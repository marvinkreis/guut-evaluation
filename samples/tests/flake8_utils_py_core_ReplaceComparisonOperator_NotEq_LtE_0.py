from flake8.utils import parse_unified_diff


def test():
    mapping = parse_unified_diff('''diff --git a/src/main/java/org/codedefenders/persistence/database/GameRepository.java b/src/main/java/org/codedefenders/persistence/database/GameRepository.java
index 211a1b729..f8af26b9b 100644
--- a/src/main/java/org/codedefenders/persistence/database/GameRepository.java
+++ b/src/main/java/org/codedefenders/persistence/database/GameRepository.java
@@ -161,11 +161,11 @@ public class GameRepository {
     public boolean removeUserFromGame(int gameId, int userId) {
         @Language("SQL") String query = """
                 UPDATE players
                 SET Active = FALSE
                 WHERE Game_ID = ?
-                  AND User_ID = ?;
+                  AND User_ID <> ?;
         """;

         int updatedRows = queryRunner.update(query, gameId, userId);
         return updatedRows > 0;
@@ -253,11 +253,11 @@ public class GameRepository {

     public boolean storeStartTime(int gameId) {
         @Language("SQL") String query = """
                 UPDATE games
                 SET Start_Time = NOW()
-                WHERE ID = ?
+                WHERE ID <> ?
         """;

         int updatedRows = queryRunner.update(query, gameId);
         return updatedRows > 0;
     }
''')
    assert 255 not in mapping["src/main/java/org/codedefenders/persistence/database/GameRepository.java"]
