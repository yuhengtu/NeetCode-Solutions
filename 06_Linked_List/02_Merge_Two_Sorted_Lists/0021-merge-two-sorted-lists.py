"""
Problem: LeetCode 21 - Merge Two Sorted Lists

Key Idea:
To merge two sorted linked lists 'l1' and 'l2', we can create a new linked list 'dummy' to hold the merged result. We maintain two pointers, 'current' and 'prev', to traverse through the two input lists. At each step, we compare the values at the 'current' pointers of 'l1' and 'l2', and add the smaller value to the 'dummy' list. We then move the 'current' pointer of the list with the smaller value one step forward. After iterating through both lists, if any list still has remaining elements, we append them to the 'dummy' list.

Time Complexity:
The time complexity of this solution is O(n), where n is the total number of nodes in the merged list. We traverse each node once to merge the lists.

Space Complexity:
The space complexity is O(1). We do not allocate new nodes. We reuse the existing nodes and just rearrange links.
"""

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        current = dummy

        while list1 and list2:
            if list1.val <= list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next
        
        if list1:
            current.next = list1
        elif list2:
            current.next = list2
        
        return dummy.next
