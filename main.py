from environment import GridEnv
from video import draw_environment, grid_to_rgb, array_to_image, images_to_video

grid = GridEnv(grid_size=40)

grid.train(500)

image_list = []

for snapshot in grid.memory:
    g1 = draw_environment(grid.world, snapshot)
    g2 = grid_to_rgb(g1)
    g3 = array_to_image(g2)
    image_list.append(g3)
images_to_video(image_list)
