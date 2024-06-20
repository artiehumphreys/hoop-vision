import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from homography.homography_calculator import HomographyCalculator
from pre_processing.image_loader import ImageLoader
import numpy as np


class CourtDrawer:
    def __init__(self, color="black", lw=2):
        self.color = color
        self.lw = lw
        self.path = "/Users/artiehumphreys/Desktop/Object Detection/full-court.jpeg"
        self.img = ImageLoader(self.path).load_image()
        self.homography_calculator = HomographyCalculator()
        self.right_bounds = np.array(
            [
                # [398, 42],  # LEFT BOTTOM
                # [752, 420],  # TOP RIGHT  (4 o'clock)
                # [752, 42],  # TOP LEFT (7 o'clock)
                # [398, 420],  # RIGHT BOTTOM
                [40, 42],  # LEFT BOTTOM
                [398, 420],  # TOP RIGHT  (4 o'clock)
                [398, 42],  # TOP LEFT (7 o'clock)
                [40, 420],  # RIGHT BOTTOM
            ]
        )

    def draw_basketball_court(self):
        fig, ax = plt.subplots(figsize=(12.8, 7.2))

        hoop = Circle(
            (250, 50), radius=7.5, linewidth=self.lw, color=self.color, fill=False
        )
        backboard = Rectangle((220, 42.5), 60, -1, linewidth=self.lw, color=self.color)
        outer_box = Rectangle(
            (170, 2.5), 160, 190, linewidth=self.lw, color=self.color, fill=False
        )
        inner_box = Rectangle(
            (190, 2.5), 120, 190, linewidth=self.lw, color=self.color, fill=False
        )

        top_free_throw = Arc(
            (250, 192.5),
            120,
            120,
            theta1=0,
            theta2=180,
            linewidth=self.lw,
            color=self.color,
            fill=False,
        )
        bottom_free_throw = Arc(
            (250, 192.5),
            120,
            120,
            theta1=180,
            theta2=0,
            linewidth=self.lw,
            color=self.color,
            linestyle="dashed",
        )
        restricted = Arc(
            (250, 50), 80, 80, theta1=0, theta2=180, linewidth=self.lw, color=self.color
        )

        corner_three_a = Rectangle(
            (30, 2.5), 0, 140, linewidth=self.lw, color=self.color
        )
        corner_three_b = Rectangle(
            (470, 2.5), 0, 140, linewidth=self.lw, color=self.color
        )
        three_arc = Arc(
            (250, 50),
            475,
            475,
            theta1=22,
            theta2=158,
            linewidth=self.lw,
            color=self.color,
        )

        center_outer_arc = Arc(
            (250, 472.5),
            120,
            120,
            theta1=180,
            theta2=0,
            linewidth=self.lw,
            color=self.color,
        )
        center_inner_arc = Arc(
            (250, 472.5),
            40,
            40,
            theta1=180,
            theta2=0,
            linewidth=self.lw,
            color=self.color,
        )

        court_elements = [
            hoop,
            backboard,
            outer_box,
            inner_box,
            top_free_throw,
            bottom_free_throw,
            restricted,
            corner_three_a,
            corner_three_b,
            three_arc,
            center_outer_arc,
            center_inner_arc,
        ]

        outer_lines = Rectangle(
            (0, 2.5), 500, 470, linewidth=self.lw, color=self.color, fill=False
        )
        court_elements.append(outer_lines)

        for element in court_elements:
            ax.add_patch(element)

        plt.xlim(0, 500)
        plt.ylim(0, 475)
        plt.axis("on")
        plt.gca().set_aspect("equal", adjustable="box")

        return ax

    def plot_transformed_positions(
        self, player_positions, court_corners, camera_corners
    ):
        coordinates = [pos for pos, _ in player_positions]
        teams = [team for _, team in player_positions]

        H = self.homography_calculator.calculate_homography_from_points(
            camera_corners, court_corners
        )
        transformed_positions = self.homography_calculator.apply_homography(
            H, coordinates
        )
        img = plt.imread(self.path)
        fig, ax = plt.subplots()
        for pos, team in zip(transformed_positions, teams):
            ax.plot(pos[0], pos[1], "o", markersize=10, color="blue")
            ax.text(pos[0], pos[1] + 2, team, color="blue")

        ax.imshow(img)
        plt.show()
