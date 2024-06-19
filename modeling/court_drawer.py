import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from homography.homography_calculator import HomographyCalculator


class CourtDrawer:
    def __init__(self, color="black", lw=2):
        self.color = color
        self.lw = lw
        self.homography_calculator = HomographyCalculator()

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
        ax = self.draw_basketball_court()
        H = self.homography_calculator.calculate_homography(
            camera_corners, court_corners
        )
        transformed_positions = self.homography_calculator.apply_homography(
            H, player_positions
        )
        for pos in transformed_positions:
            plt.plot(pos[0], pos[1], "o", markersize=10, color="blue")

        plt.show()
