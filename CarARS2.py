# Car simulator class from course ARS2

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter


class CarARS2:
    """Simulated car"""

    def __init__(self):
        # Vehicle state
        self.reset_state()

        # Lane info
        self.lane_width = 6
        self.marking_width = 0.2

        # Vehicle valid state
        self.x_min, self.x_max = 0, 300 
        self.y_min, self.y_max = -self.lane_width / 2, self.lane_width / 2
        self.theta_min, self.theta_max = -np.pi / 12, np.pi / 12

        # Camera parameters
        self.img_w = 640
        self.img_h = 480
        self.cam_fu = 25
        self.cam_fv = 85
        self.cam_cu = self.img_w // 2
        self.cam_cv = self.img_h // 2

        # Animation parameters
        self.is_pause = False
        self.render = True
        self.freq = 10  # Animation frequency in Hz
        #
        if self.render:
            self.update_cam()

    def update_state(self):
        """
        Update vehicle state with constant velocity
        """
        dtheta = self.theta_dot * 1 / self.freq
        self.x += self.v * np.cos(self.theta + dtheta / 2)
        self.y += self.v * np.sin(self.theta + dtheta / 2)
        self.theta += dtheta
        self.t += 1 / self.freq
        if self.render:
            self.update_cam()

    def reset_state(self):
        """
        Reset vehicle state
        """
        # Vehicle state
        self.x = 0  # East coordinates in meter
        self.y = 0  # North coordinates in meter
        self.theta = (np.random.rand() - 0.5) * np.pi / 32  # Heading in radian
        self.theta_dot = 0
        self.v = 1  # Velocity in meter per second
        self.t = 0  # Time in second

    def cam_coord(self, Yw, Zc):
        # Image coordinates of point depth Zc along the line y=Yw in the world frame
        u = (
            self.cam_fu
            * (np.tan(self.theta) + (self.y - Yw) / (Zc * np.cos(self.theta)))
            + self.cam_cu
        )
        v = self.cam_fv / Zc + self.cam_cv
        return u, v

    def vanish_coord(self):
        # Image coordinates of the vanishing point
        return self.cam_fu * np.tan(self.theta) + self.cam_cu, self.cam_cv

    def update_cam(self):
        self.img = Image.new("L", (self.img_w, self.img_h))
        imgd = ImageDraw.Draw(self.img)
        for y in [self.y_min, self.y_max]:
            imgd.polygon(
                [
                    self.vanish_coord(),
                    self.cam_coord(
                        y - self.marking_width / 2,
                        self.cam_fv / (self.img_h - self.cam_cv),
                    ),
                    self.cam_coord(
                        y + self.marking_width / 2,
                        self.cam_fv / (self.img_h - self.cam_cv),
                    ),
                ],
                fill="white",
            )
        self.img = self.img.filter(ImageFilter.GaussianBlur(radius=1))

    def valid_state(self):
        return (
            self.x <= self.x_max
            and self.x >= self.x_min
            and self.y <= self.y_max
            and self.y >= self.y_min
            and self.theta <= self.theta_max
            and self.theta >= self.theta_min
        )

    def turn_left(self, dtheta=np.pi / 180, sd=0.005):
        self.theta += dtheta + np.random.normal(scale=sd)

    def turn_right(self, dtheta=np.pi / 180, sd=0.005):
        self.theta -= dtheta + np.random.normal(scale=sd)

    def pause(self):
        self.is_pause = not self.is_pause

    def press(self, event):
        if event.key == "left":
            self.turn_left()
        elif event.key == "right":
            self.turn_right()
        elif event.key == " ":
            self.pause()

    def display_init(self):
        self.fig, self.axes = plt.subplots(2, 2)
        self.fig.canvas.mpl_connect("key_press_event", self.press)

        self.top, self.cam, self.err_lat, self.err_ang = (
            self.axes[0, 0],
            self.axes[1, 0],
            self.axes[0, 1],
            self.axes[1, 1],
        )
        self.top.set_title("Top-down view")
        self.cam.set_title("Camera image")
        self.err_lat.set_title("Lateral error")
        self.err_ang.set_title("Angular error")

        self.cam.set_xlim([0, self.img_w])
        self.cam.set_ylim([self.img_h, 0])
        self.err_lat.set_ylim([self.y_min, self.y_max])
        self.err_ang.set_ylim([self.theta_min, self.theta_max])

        # Draw road in top-down view
        self.top.set_ylim([self.y_min - 1, self.y_max + 1])
        self.top.plot(
            [self.x_min, self.x_max], [self.y_min, self.y_min], "k-", linewidth=3
        )
        self.top.plot(
            [self.x_min, self.x_max], [self.y_max, self.y_max], "k-", linewidth=3
        )
        zx = np.arange(self.x_min, self.x_max + 10, 10)
        zy = np.tile(np.array([self.y_max, self.y_min]), len(zx) // 2)
        if len(zx) % 2:
            zy = np.concatenate((zy, np.array([-zy[-1]])))
        self.top.plot(zx, zy, color="0.5")

        # Display elements
        self.arrow = None
        self.frame = None

    def display_update(self):
        try:
            self.arrow.remove()
        except:
            pass
        try:
            self.frame.remove()
        except:
            pass
        # Plot vehicle on the road
        self.arrow = self.top.arrow(
            self.x, self.y, np.cos(self.theta), np.sin(self.theta), width=0.1
        )
        self.top.set_xlim([self.x - 4, self.x + 4])
        # Show camera image
        self.frame = self.cam.imshow(self.img, cmap="gray")
        # Error in lateral direction
        self.err_lat.plot(self.t, self.y, "k+")
        self.err_ang.plot(self.t, self.theta, "k+")


if __name__ == "__main__":
    car = CarARS2()
    car.display_init()

    while car.valid_state():
        car.display_update()
        # Pause
        if not car.is_pause:
            car.update_state()
        plt.pause(1 / car.freq)

    plt.show()
