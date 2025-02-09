from environment import GridEnv
from video import draw_environment, grid_to_rgb, array_to_image, images_to_video

i = 0
while True:
    grid = GridEnv(
        fov=2,
        grid_size=160,
        learning_rate=0.9,
        discount_factor=0.94,
        exploration_rate=0.2,
        # num_agent=10,
        # fov=3,
        # grid_size=200,
        # learning_rate=0.9,
        # discount_factor=0.99,
        # exploration_rate=0.3,
    )

    grid.train(1000)
    # grid.simulate(500, filename="agent_bk2.pkl")

    image_list = []

    for snapshot in grid.memory:
        g1 = draw_environment(grid.world, snapshot)
        g2 = grid_to_rgb(g1)
        g3 = array_to_image(g2)
        image_list.append(g3)
    images_to_video(image_list, filename=f"training.avi")

    grid = GridEnv(
        fov=2,
        grid_size=160,
        learning_rate=0.9,
        discount_factor=0.94,
        exploration_rate=0.01,
        # num_agent=10,
        # fov=3,
        # grid_size=200,
        # learning_rate=0.9,
        # discount_factor=0.99,
        # exploration_rate=0.3,
    )

    grid.simulate(1000)

    image_list = []
    for snapshot in grid.memory:
        g1 = draw_environment(grid.world, snapshot)
        g2 = grid_to_rgb(g1)
        g3 = array_to_image(g2)
        image_list.append(g3)
    images_to_video(image_list, filename=f"simulating.avi")
    quit()

    if i % 20 == 0:
        for snapshot in grid.memory:
            g1 = draw_environment(grid.world, snapshot)
            g2 = grid_to_rgb(g1)
            g3 = array_to_image(g2)
            image_list.append(g3)
        images_to_video(image_list, filename=f"output-{i}.avi")
        print(f"output-{i}.avi")
        i += 1
