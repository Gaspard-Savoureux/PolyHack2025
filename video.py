#!/usr/bin/env python

import numpy as np
import cv2 as cv
import random

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

    return array

def pixel_to_rgb(pixel):
    match pixel:
        case 1:
            return np.array([0, 0, 0])
        case 0:
            return np.array([255, 255, 255])
        case _:
            return np.array([0, 0, 0])


def grid_to_rgb(grid):
    rgb_array = []

    for row in grid:
        new_row = []
        for pixel in row:
            rgb_pixel = pixel_to_rgb(pixel)
            new_row.append(rgb_pixel)
        rgb_array.append(new_row)

    return np.array(rgb_array)


def array_to_images(grid, frame_size):
    normalized_grid = cv.normalize(grid, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    image = cv.cvtColor(normalized_grid, cv.COLOR_RGB2BGR)
    scaled_image = cv.resize(image, frame_size, interpolation=cv.INTER_NEAREST)
    return scaled_image


def images_to_video(images, frame_size):
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    video = cv.VideoWriter('output.mp4', fourcc, 1, frame_size)

    for image in images:
        video.write(image)

    video.release()


if __name__ == '__main__':
    frame_size = (1000, 1000)
    rows, cols = 1000, 1000
    fill_ratio = 0.6
    num_blobs = 5

    grid = generate_blobs(rows, cols, fill_ratio, num_blobs)
    rgb_grid = grid_to_rgb(grid)
    image = array_to_images(rgb_grid, frame_size)

    images_to_video([image, image, image], frame_size)

    # image = np.hstack((image, image))

    # cv.imshow('test', image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
