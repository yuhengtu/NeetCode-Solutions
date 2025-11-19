# time: O(m + n); space: O(1), 26 English characters

class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        magazine_count = {}
        for m in magazine:
            magazine_count[m] = magazine_count.get(m, 0) + 1
        
        for r in ransomNote:
            if r not in magazine_count or magazine_count[r] == 0:
                return False
            magazine_count[r] -= 1

        return True
