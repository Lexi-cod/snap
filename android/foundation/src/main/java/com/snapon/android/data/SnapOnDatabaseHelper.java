package com.snapon.android.data;

import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

final class SnapOnDatabaseHelper extends SQLiteOpenHelper {
    static final String DATABASE_NAME = "snapon_memory.db";
    static final int DATABASE_VERSION = 2;

    static final String TABLE_MEMORIES = "memories";
    static final String TABLE_SETTINGS = "settings";

    SnapOnDatabaseHelper(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        db.execSQL(
                "CREATE TABLE IF NOT EXISTS " + TABLE_MEMORIES + " ("
                        + "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                        + "text TEXT NOT NULL,"
                        + "tag TEXT NOT NULL DEFAULT '',"
                        + "created_at_epoch_millis INTEGER NOT NULL,"
                        + "access_count INTEGER NOT NULL DEFAULT 0,"
                        + "image_uri TEXT,"
                        + "image_description TEXT,"
                        + "embedding BLOB"
                        + ")"
        );
        db.execSQL(
                "CREATE TABLE IF NOT EXISTS " + TABLE_SETTINGS + " ("
                        + "key TEXT PRIMARY KEY,"
                        + "value TEXT NOT NULL"
                        + ")"
        );
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_memories_created_at ON "
                + TABLE_MEMORIES + "(created_at_epoch_millis DESC)");
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_memories_tag ON "
                + TABLE_MEMORIES + "(tag)");
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        if (oldVersion < 1) {
            onCreate(db);
            return;
        }
        if (oldVersion < 2) {
            db.execSQL("ALTER TABLE " + TABLE_MEMORIES + " ADD COLUMN embedding BLOB");
        }
    }
}
