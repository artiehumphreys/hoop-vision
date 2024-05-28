import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from homography import calculate_points as cp


# http://savvastjortjoglou.com/nba-shot-sharts.html
def draw_basketball_court(color="black", lw=2):
    ax = plt.gca()
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False)
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False)

    top_free_throw = Arc(
        (0, 142.5),
        120,
        120,
        theta1=0,
        theta2=180,
        linewidth=lw,
        color=color,
        fill=False,
    )
    bottom_free_throw = Arc(
        (0, 142.5),
        120,
        120,
        theta1=180,
        theta2=0,
        linewidth=lw,
        color=color,
        linestyle="dashed",
    )
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color)

    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw, color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)

    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color)

    center_outer_arc = Arc(
        (0, 422.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color
    )
    center_inner_arc = Arc(
        (0, 422.5), 40, 40, theta1=180, theta2=0, linewidth=lw, color=color
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
        (-250, -47.5), 500, 470, linewidth=lw, color=color, fill=False
    )
    court_elements.append(outer_lines)

    for element in court_elements:
        ax.add_patch(element)

    plt.xlim(-250, 250)
    plt.ylim(-50, 470)
    plt.axis("off")

    return ax


def plot_transformed_positions(player_positions, court_corners, camera_corners):
    ax = draw_basketball_court()
    H = cp.calculate_homography(camera_corners, court_corners)
    transformed_positions = cp.apply_homography(H, player_positions)
    for pos in transformed_positions:
        plt.plot(pos[0], pos[1], "o", markersize=10, color="blue")

    plt.show()
