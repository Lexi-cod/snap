package com.snapon.android.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class MemorySearch {
    private static final Pattern WORD = Pattern.compile("\\b\\w+\\b");
    private static final Set<String> STOPWORDS = new HashSet<>(Arrays.asList(
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "i", "my", "your", "their", "our", "its", "this", "that", "these", "those",
            "what", "where", "when", "who", "how", "why", "which",
            "in", "on", "at", "to", "of", "for", "with", "by", "from", "up", "about", "into",
            "and", "or", "but", "if", "as", "so", "yet"
    ));

    private MemorySearch() {
    }

    public static double keywordScore(String query, String memoryText) {
        List<String> queryWords = keywords(query);
        if (queryWords.isEmpty()) {
            return 0.0;
        }

        Set<String> memoryWords = new HashSet<>(keywords(memoryText));
        double matched = 0.0;
        for (String queryWord : queryWords) {
            if (memoryWords.contains(queryWord)) {
                matched += 1.0;
            } else if (queryWord.length() >= 4 && hasPrefix(memoryWords, queryWord.substring(0, 4))) {
                matched += 0.8;
            }
        }
        return matched / queryWords.size();
    }

    private static boolean hasPrefix(Set<String> words, String prefix) {
        for (String word : words) {
            if (word.startsWith(prefix)) {
                return true;
            }
        }
        return false;
    }

    private static List<String> keywords(String value) {
        List<String> words = new ArrayList<>();
        if (value == null) {
            return words;
        }

        Matcher matcher = WORD.matcher(value.toLowerCase(Locale.US));
        while (matcher.find()) {
            String word = matcher.group();
            if (word.length() > 2 && !STOPWORDS.contains(word)) {
                words.add(word);
            }
        }
        return words;
    }
}
