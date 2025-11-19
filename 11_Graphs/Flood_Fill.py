# Time Complexity: O(m × n); Space Complexity: O(m × n)
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        org_color = image[sr][sc]
        if org_color == color:
            return image
        
        n_rows, n_cols = len(image), len(image[0])
        def dfs(r, c, org_color):
            image[r][c] = color
            directions = [[1,0], [-1,0], [0,1], [0,-1]]
            for (r_diff, c_diff) in directions:
                r_nxt, c_nxt = r+r_diff, c+c_diff
                if (
                    0 <= r_nxt < n_rows
                    and 0 <= c_nxt < n_cols
                    and image[r_nxt][c_nxt] == org_color
                ):
                    dfs(r_nxt, c_nxt, org_color)

        dfs(sr, sc, org_color)
        return image
      
