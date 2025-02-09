#!/usr/bin/env python

import numpy as np
import cv2 as cv
import random

# generation settings
ROWS = 100
COLS = 100
FILL_RATIO = 0.2
NUM_BLOBS = 20

# video settings
CODEC = 'XVID'
FRAMES_PER_SECOND = 1

# image settings
FRAME_SIZE = (1000, 1000)  # (width, height)

# RGB colors
MINERAL_COLOR = np.array([50, 50, 50])
EMPTY_COLOR = np.array([150, 150, 150])
ERROR_COLOR = np.array([255, 0, 0])
AGENT = np.array([0, 255, 255])
DISCOVERED_EMPTY = np.array([100, 100, 100])
JUST_DISCOVERED_EMPTY = np.array([50, 50, 50])
DISCOVERED_MINERAL = np.array([255, 255, 0])
JUST_DISCOVERED_MINERAL = np.array([200, 200, 0])


def generate_blobs(rows, cols, fill_ratio, num_blobs):
    """
    Generates a 2D array with multiple random blobs while ensuring a specific fill ratio.

    Parameters:
    - rows (int): Number of rows in the grid.
    - cols (int): Number of columns in the grid.
    - fill_ratio (float): Target ratio of filled cells (0 to 1).
    - num_blobs (int): Number of distinct blobs.

    Returns:
    - np.array: A 2D numpy array with 0s (empty) and 1s (filled blobs).
    """
    grid_size = rows * cols
    target_fill = int(grid_size * fill_ratio)
    
    # Initialize grid
    array = np.zeros((rows, cols), dtype=int)

    # Generate unique starting positions for each blob
    blob_positions = set()
    while len(blob_positions) < num_blobs:
        blob_positions.add((random.randint(0, rows - 1), random.randint(0, cols - 1)))

    filled_cells = 0

    # Function to perform random walk from a given position
    def random_walk(x, y, steps):
        nonlocal filled_cells
        for _ in range(steps):
            if filled_cells >= target_fill:
                return
            if array[x, y] == 0:  # Avoid double counting
                array[x, y] = 1
                filled_cells += 1

            # Move randomly in four directions
            dx, dy = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
            x, y = max(0, min(rows - 1, x + dx)), max(0, min(cols - 1, y + dy))  # Stay in bound

    # Distribute steps among blobs
    steps_per_blob = max(1, target_fill // num_blobs)

    # Perform random walks for each blob
    for x, y in blob_positions:
        random_walk(x, y, steps_per_blob)

    available_steps = []
    for x in range(rows):
        for y in range(cols):
            if array[x, y] == 1:
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < rows and 0 <= new_y < cols and array[new_x, new_y] == 0:
                        available_steps.append((new_x, new_y))

    for _ in range(target_fill):
        x, y = random.choice(available_steps)
        array[x, y] = 1
        available_steps.remove((x, y))
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < rows and 0 <= new_y < cols:
                available_steps.append((new_x, new_y))

    return array


def pixel_to_rgb(pixel):
    match pixel:
        case 0:
            return EMPTY_COLOR
        case 1:
            return MINERAL_COLOR
        case 2:
            return DISCOVERED_EMPTY
        case 3:
            return JUST_DISCOVERED_EMPTY
        case 4:
            return DISCOVERED_MINERAL
        case 5:
            return JUST_DISCOVERED_MINERAL
        case _:
            return ERROR_COLOR


def grid_to_rgb(grid):
    return np.array([pixel_to_rgb(pixel) for pixel in grid.flatten()]).reshape(grid.shape[0], grid.shape[1], 3)


def array_to_image(grid):
    grid = np.array(grid, dtype=np.uint8)
    image = cv.cvtColor(grid, cv.COLOR_RGB2BGR)
    scaled_image = cv.resize(image, FRAME_SIZE, interpolation=cv.INTER_NEAREST)
    return scaled_image


def images_to_video(images):
    codec = cv.VideoWriter_fourcc(*CODEC)
    video = cv.VideoWriter('output.mp4', codec, FRAMES_PER_SECOND, FRAME_SIZE)

    for image in images:
        video.write(image)

    video.release()


def draw_environment(grid, grid_env_memory):
    for (x, y) in grid_env_memory.agents:
        grid[x][y] = AGENT
    for (x, y) in grid_env_memory.discovered_empty:
        grid[x][y] = DISCOVERED_EMPTY
    for (x, y) in grid_env_memory.just_discovered_empty:
        grid[x][y] = JUST_DISCOVERED_EMPTY
    for (x, y) in grid_env_memory.discovered_vein:
        grid[x][y] = DISCOVERED_MINERAL
    for (x, y) in grid_env_memory.just_discovered_vein:
        grid[x][y] = JUST_DISCOVERED_MINERAL

    return grid
    

if __name__ == '__main__':
    
    generated_map = generate_blobs(ROWS, COLS, FILL_RATIO, NUM_BLOBS)
    
    image = array_to_image(grid_to_rgb(generated_map))

    # images_to_video([image, image, image], frame_size)

    # image = np.hstack((image, image))

    cv.imshow('test', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
