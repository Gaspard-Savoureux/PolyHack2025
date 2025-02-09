from environment import GridEnv
from video import draw_environment, grid_to_rgb, array_to_image, images_to_video

grid = GridEnv(
    # fov=2,
    # grid_size=40,
    # learning_rate=0.9,
    # discount_factor=0.99,
    # exploration_rate=0.01,
    fov=2,
    grid_size=100,
    learning_rate=0.9,
    discount_factor=0.99,
    exploration_rate=0.3,
)

grid.train(1000)
# grid.simulate(500)

image_list = []

for snapshot in grid.memory:
    g1 = draw_environment(grid.world, snapshot)
    g2 = grid_to_rgb(g1)
    g3 = array_to_image(g2)
    image_list.append(g3)
images_to_video(image_list)
