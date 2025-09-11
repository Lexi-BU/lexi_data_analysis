from moviepy import VideoFileClip, clips_array

gif_folder = "../movies/"
# Load the two GIFs
gif1 = VideoFileClip("output_video_2025-03-06_14-00-00_2025-03-06_17-00-00.gif")
gif2 = VideoFileClip("output_video_2025-03-08_02-00-00_2025-03-08_05-00-00.gif")

# Resize the GIFs to have the same height (optional)
# This ensures they align properly when combined
if gif1.size[1] != gif2.size[1]:
    target_height = min(gif1.size[1], gif2.size[1])
    gif1 = gif1.resize(height=target_height)
    gif2 = gif2.resize(height=target_height)

# Combine the GIFs side by side
combined_clip = clips_array([[gif1, gif2]])

# Save the combined GIF
combined_clip.write_gif("combined_side_by_side.gif", fps=gif1.fps)

print("Combined GIF saved as combined_side_by_side.gif")
