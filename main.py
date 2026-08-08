import numpy as np
import pygame
from window.draw_window import Window
from window.NN import NN

np.set_printoptions(suppress=True, precision=3)
params = np.load("weights.npz")
BLACK = (0, 0, 0)
GREEN = (0, 234, 0)


pygame.init()
pygame.font.init()


font = pygame.font.Font(None, 40)
screen = pygame.display.set_mode((602, 800))
pygame.display.set_caption("Digit Recogniser")

window = Window(screen)
nn = NN(screen, list(dict(params).values()))
done = False

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if (
                event.key == pygame.K_RETURN
            ):  # nn will predict the number with probability
                if not done:
                    done = True
                    prediction, pred_prob = nn.predict(
                        window.grid.flatten().reshape(-1, 1) / 255
                    )
                    for pred, prob in zip(prediction, pred_prob):
                        print(f"{pred} ---> {prob}")
                else:
                    done = False

            if event.key == pygame.K_r:  # reload the window, new window
                window.clear()
                if done:
                    done = False

    screen.fill(BLACK)
    window.make_window()
    window.draw_pixels()
    if not done:
        window.draw()
    else:
        text1 = font.render(f"Prediction: {prediction[0][0]}", True, GREEN)
        text2 = font.render(f"Probability: {pred_prob[0][0]:.2f}%", True, GREEN)

        screen.blit(text1, (180, 650))
        screen.blit(text2, (180, 700))

    pygame.display.flip()

pygame.quit()
