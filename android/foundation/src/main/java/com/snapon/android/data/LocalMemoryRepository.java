package com.snapon.android.data;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public final class LocalMemoryRepository implements MemoryRepository {
    private static final double KEYWORD_THRESHOLD = 0.25;

    private final SnapOnDatabaseHelper databaseHelper;
    private final TimeSource timeSource;

    public LocalMemoryRepository(Context context) {
        this(context, System::currentTimeMillis);
    }

    LocalMemoryRepository(Context context, TimeSource timeSource) {
        this.databaseHelper = new SnapOnDatabaseHelper(context.getApplicationContext());
        this.timeSource = timeSource;
    }

    @Override
    public Memory save(MemoryDraft draft) throws IOException {
        SQLiteDatabase db = databaseHelper.getWritableDatabase();
        db.beginTransaction();
        try {
            SnapOnSettings settings = loadSettingsLocked(db);
            int count = countLocked(db);
            if (count >= settings.getMaxMemories()) {
                if (!settings.isAutoCleanup()) {
                    throw new IOException("memory limit reached");
                }
                deleteOldestLocked(db);
            }

            ContentValues values = new ContentValues();
            values.put("text", storedText(draft));
            values.put("tag", draft.getTag());
            values.put("created_at_epoch_millis", timeSource.now());
            values.put("access_count", 0);
            values.put("image_uri", draft.getImageUri());
            values.put("image_description", draft.getImageDescription());
            long id = db.insertOrThrow(SnapOnDatabaseHelper.TABLE_MEMORIES, null, values);
            db.setTransactionSuccessful();
            return getLocked(db, id);
        } finally {
            db.endTransaction();
        }
    }

    @Override
    public List<Memory> list() {
        SQLiteDatabase db = databaseHelper.getReadableDatabase();
        try (Cursor cursor = db.query(
                SnapOnDatabaseHelper.TABLE_MEMORIES,
                null,
                null,
                null,
                null,
                null,
                "id DESC"
        )) {
            return readMemories(cursor);
        }
    }

    @Override
    public Memory get(long id) throws IOException {
        SQLiteDatabase db = databaseHelper.getReadableDatabase();
        Memory memory = getLocked(db, id);
        if (memory == null) {
            throw new IOException("memory not found: " + id);
        }
        return memory;
    }

    @Override
    public boolean delete(long id) {
        SQLiteDatabase db = databaseHelper.getWritableDatabase();
        return db.delete(SnapOnDatabaseHelper.TABLE_MEMORIES, "id = ?", new String[]{String.valueOf(id)}) > 0;
    }

    @Override
    public void deleteAll() {
        SQLiteDatabase db = databaseHelper.getWritableDatabase();
        db.delete(SnapOnDatabaseHelper.TABLE_MEMORIES, null, null);
    }

    @Override
    public List<Memory> retrieve(String query, int limit) {
        int safeLimit = Math.max(1, limit);
        SQLiteDatabase db = databaseHelper.getWritableDatabase();
        List<ScoredMemory> scored = new ArrayList<>();

        try (Cursor cursor = db.query(
                SnapOnDatabaseHelper.TABLE_MEMORIES,
                null,
                null,
                null,
                null,
                null,
                "id ASC"
        )) {
            while (cursor.moveToNext()) {
                Memory memory = readMemory(cursor);
                double score = MemorySearch.keywordScore(query, searchText(memory));
                if (score >= KEYWORD_THRESHOLD) {
                    scored.add(new ScoredMemory(memory, score));
                }
            }
        }

        Collections.sort(scored, new Comparator<ScoredMemory>() {
            @Override
            public int compare(ScoredMemory left, ScoredMemory right) {
                int byScore = Double.compare(right.score, left.score);
                if (byScore != 0) {
                    return byScore;
                }
                return Long.compare(right.memory.getId(), left.memory.getId());
            }
        });

        List<Memory> results = new ArrayList<>();
        for (int i = 0; i < scored.size() && i < safeLimit; i++) {
            Memory memory = scored.get(i).memory;
            incrementAccessCount(db, memory.getId());
            results.add(memory);
        }
        return results;
    }

    @Override
    public SnapOnSettings loadSettings() {
        return loadSettingsLocked(databaseHelper.getReadableDatabase());
    }

    @Override
    public void saveSettings(SnapOnSettings settings) {
        SQLiteDatabase db = databaseHelper.getWritableDatabase();
        db.beginTransaction();
        try {
            putSettingLocked(db, "max_memories", String.valueOf(settings.getMaxMemories()));
            putSettingLocked(db, "auto_cleanup", String.valueOf(settings.isAutoCleanup()));
            putSettingLocked(db, "similarity_threshold", String.valueOf(settings.getSimilarityThreshold()));
            putSettingLocked(db, "visual_threshold", String.valueOf(settings.getVisualThreshold()));
            db.setTransactionSuccessful();
        } finally {
            db.endTransaction();
        }
    }

    @Override
    public void close() {
        databaseHelper.close();
    }

    private static String storedText(MemoryDraft draft) {
        if (draft.getImageDescription() == null) {
            return draft.getText();
        }
        return draft.getText() + "\n[Visual: " + draft.getImageDescription() + "]";
    }

    private static String searchText(Memory memory) {
        StringBuilder builder = new StringBuilder(memory.getText());
        if (memory.getImageDescription() != null) {
            builder.append(' ').append(memory.getImageDescription());
        }
        if (!memory.getTag().isEmpty()) {
            builder.append(' ').append(memory.getTag());
        }
        return builder.toString();
    }

    private int countLocked(SQLiteDatabase db) {
        try (Cursor cursor = db.rawQuery(
                "SELECT COUNT(*) FROM " + SnapOnDatabaseHelper.TABLE_MEMORIES,
                null
        )) {
            cursor.moveToFirst();
            return cursor.getInt(0);
        }
    }

    private void deleteOldestLocked(SQLiteDatabase db) {
        db.execSQL("DELETE FROM " + SnapOnDatabaseHelper.TABLE_MEMORIES
                + " WHERE id = (SELECT id FROM " + SnapOnDatabaseHelper.TABLE_MEMORIES
                + " ORDER BY id ASC LIMIT 1)");
    }

    private Memory getLocked(SQLiteDatabase db, long id) {
        try (Cursor cursor = db.query(
                SnapOnDatabaseHelper.TABLE_MEMORIES,
                null,
                "id = ?",
                new String[]{String.valueOf(id)},
                null,
                null,
                null
        )) {
            return cursor.moveToFirst() ? readMemory(cursor) : null;
        }
    }

    private static void incrementAccessCount(SQLiteDatabase db, long id) {
        db.execSQL(
                "UPDATE " + SnapOnDatabaseHelper.TABLE_MEMORIES
                        + " SET access_count = access_count + 1 WHERE id = ?",
                new Object[]{id}
        );
    }

    private static List<Memory> readMemories(Cursor cursor) {
        List<Memory> memories = new ArrayList<>();
        while (cursor.moveToNext()) {
            memories.add(readMemory(cursor));
        }
        return memories;
    }

    private static Memory readMemory(Cursor cursor) {
        return new Memory(
                cursor.getLong(cursor.getColumnIndexOrThrow("id")),
                cursor.getString(cursor.getColumnIndexOrThrow("text")),
                cursor.getString(cursor.getColumnIndexOrThrow("tag")),
                cursor.getLong(cursor.getColumnIndexOrThrow("created_at_epoch_millis")),
                cursor.getInt(cursor.getColumnIndexOrThrow("access_count")),
                cursor.getString(cursor.getColumnIndexOrThrow("image_uri")),
                cursor.getString(cursor.getColumnIndexOrThrow("image_description"))
        );
    }

    private static SnapOnSettings loadSettingsLocked(SQLiteDatabase db) {
        SnapOnSettings defaults = SnapOnSettings.defaults();
        int maxMemories = getIntSetting(db, "max_memories", defaults.getMaxMemories());
        boolean autoCleanup = getBooleanSetting(db, "auto_cleanup", defaults.isAutoCleanup());
        double similarityThreshold = getDoubleSetting(
                db,
                "similarity_threshold",
                defaults.getSimilarityThreshold()
        );
        double visualThreshold = getDoubleSetting(db, "visual_threshold", defaults.getVisualThreshold());
        return new SnapOnSettings(maxMemories, autoCleanup, similarityThreshold, visualThreshold);
    }

    private static int getIntSetting(SQLiteDatabase db, String key, int fallback) {
        String value = getSettingLocked(db, key);
        if (value == null) {
            return fallback;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private static boolean getBooleanSetting(SQLiteDatabase db, String key, boolean fallback) {
        String value = getSettingLocked(db, key);
        return value == null ? fallback : Boolean.parseBoolean(value);
    }

    private static double getDoubleSetting(SQLiteDatabase db, String key, double fallback) {
        String value = getSettingLocked(db, key);
        if (value == null) {
            return fallback;
        }
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private static String getSettingLocked(SQLiteDatabase db, String key) {
        try (Cursor cursor = db.query(
                SnapOnDatabaseHelper.TABLE_SETTINGS,
                new String[]{"value"},
                "key = ?",
                new String[]{key},
                null,
                null,
                null
        )) {
            return cursor.moveToFirst() ? cursor.getString(0) : null;
        }
    }

    private static void putSettingLocked(SQLiteDatabase db, String key, String value) {
        ContentValues values = new ContentValues();
        values.put("key", key);
        values.put("value", value);
        db.insertWithOnConflict(
                SnapOnDatabaseHelper.TABLE_SETTINGS,
                null,
                values,
                SQLiteDatabase.CONFLICT_REPLACE
        );
    }

    interface TimeSource {
        long now();
    }

    private static final class ScoredMemory {
        final Memory memory;
        final double score;

        ScoredMemory(Memory memory, double score) {
            this.memory = memory;
            this.score = score;
        }
    }
}
