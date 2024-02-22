import pygame
#
if __name__=="__main__":
    """
               Shows dots on screen center to correctly position the calibration device.
               """

    # Initialize Pygame
    pygame.init()

    # Get screen size
    screen_width, screen_height = pygame.display.Info().current_w, pygame.display.Info().current_h

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Display Calibration")

    # Set circle radius
    radius = 5

    # Calculate center coordinates
    center_x = screen_width // 2
    center_y = screen_height // 2

    # Draw red circle at center
    pygame.draw.circle(screen, (255, 0, 0), (center_x, center_y), radius)

    # Draw red circles on the left and right
    left_x = center_x - radius - 100
    right_x = center_x + radius + 100
    pygame.draw.circle(screen, (255, 0, 0), (left_x, center_y), radius)
    pygame.draw.circle(screen, (255, 0, 0), (right_x, center_y), radius)

    # Update display
    pygame.display.flip()

    # Wait for user to close window
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    # Quit Pygame
    pygame.quit()