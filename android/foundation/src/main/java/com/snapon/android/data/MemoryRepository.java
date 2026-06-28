package com.snapon.android.data;

import java.io.Closeable;
import java.io.IOException;
import java.util.List;

public interface MemoryRepository extends Closeable {
    Memory save(MemoryDraft draft) throws IOException;

    List<Memory> list() throws IOException;

    Memory get(long id) throws IOException;

    boolean delete(long id) throws IOException;

    void deleteAll() throws IOException;

    List<Memory> retrieve(String query, int limit) throws IOException;

    SnapOnSettings loadSettings() throws IOException;

    void saveSettings(SnapOnSettings settings) throws IOException;
}
