"""
Problem: LeetCode 424 - Longest Repeating Character Replacement

Key Idea:
To find the maximum length of a substring with at most k distinct characters in the input string 's', we can use the sliding window approach. We use two pointers, 'left' and 'right', to represent the current window. As we move the 'right' pointer to the right, we expand the window and add characters to a dictionary to keep track of their frequencies. If the number of distinct characters in the window exceeds k, we move the 'left' pointer to the right to shrink the window until the number of distinct characters is k again. At each step, we update the maximum length of the window.

Time Complexity:
The time complexity of this solution is O(n), where n is the length of the input string 's'. The sliding window approach iterates through the string once, and at each step, we perform constant-time operations to update the window and the maximum length.

Space Complexity:
The space complexity is O(k), where k is the maximum number of distinct characters allowed in the input string 's'. In the worst case, the entire string can have k distinct characters, so the dictionary storing the character frequencies in the window would have k elements.
"""


class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        l, r = 0, 0
        max_len = 0
        char2freq = {}
        max_freq = 0

        for r in range(len(s)):
            char2freq[s[r]] = char2freq.get(s[r], 0) + 1
            max_freq = max(max_freq, char2freq[s[r]])

            if (r-l+1) - max_freq > k:
                char2freq[s[l]] -= 1
                l += 1
            
            max_len = max(max_len, r-l+1)
        
        return max_len
