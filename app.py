import streamlit as st
import plotly.graph_objects as go
import numpy as np
from itertools import combinations
import ast


def draw_spheres(fig, centers, R):
    """Draw spheres around points using Mesh3d."""
    for x, y, z in centers:
        phi = np.linspace(0, 2 * np.pi, 30)
        theta = np.linspace(0, np.pi, 30)
        phi, theta = np.meshgrid(phi, theta)

        x_sphere = R * np.sin(theta) * np.cos(phi) + x
        y_sphere = R * np.sin(theta) * np.sin(phi) + y
        z_sphere = R * np.cos(theta) + z

        i, j, k = [], [], []
        for u in range(0, 29):
            for v in range(0, 29):
                i.append(u * 30 + v)
                j.append((u + 1) * 30 + v)
                k.append(u * 30 + v + 1)

        fig.add_trace(
            go.Mesh3d(
                x=x_sphere.flatten(),
                y=y_sphere.flatten(),
                z=z_sphere.flatten(),
                i=i,
                j=j,
                k=k,
                opacity=0.3,
                color="#808080",
            )
        )


def distance(p1, p2):
    """Calculate Euclidean distance between two points in 3D space."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def get_combinations(iterable, r):
    """Wrapper around itertools combinations to make it more readable."""
    return combinations(iterable, r)


def filter_combinations_by_distance(points, max_distance):
    """Filter combinations of points that are within a certain distance."""
    combinations_within_distance = [
        (i, j)
        for i, j in get_combinations(range(len(points)), 2)
        if distance(points[i], points[j]) < max_distance
    ]
    return combinations_within_distance


def generate_scatter_trace(centers):
    """Generate a scatter plot trace for the centers."""
    return go.Scatter3d(
        x=centers[:, 0], y=centers[:, 1], z=centers[:, 2], mode="markers"
    )


def generate_edge_trace(points, edge_indices):
    """Generate edge traces."""
    edge_x, edge_y, edge_z = [], [], []
    for i, j in edge_indices:
        p1, p2 = points[i], points[j]
        edge_x.extend([p1[0], p2[0], None])
        edge_y.extend([p1[1], p2[1], None])
        edge_z.extend([p1[2], p2[2], None])

    return go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode="lines", line=dict(color="blue", width=2)
    )


def generate_mesh_trace(centers, i, j, k, color, opacity):
    """Generate mesh traces for tetrahedra or triangles."""
    return go.Mesh3d(
        x=centers[:, 0],
        y=centers[:, 1],
        z=centers[:, 2],
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=opacity,
    )


def generate_layout(width, height):
    """Generate layout configurations."""
    return dict(
        width=width, height=height, scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    )


def filter_tetrahedra_combinations(points, max_distance):
    """Get combinations of points forming tetrahedra within a certain distance."""
    return [
        (c1, c2, c3, c4)
        for c1, c2, c3, c4 in get_combinations(range(len(points)), 4)
        if all(
            distance(points[i], points[j]) < max_distance
            for i, j in get_combinations([c1, c2, c3, c4], 2)
        )
    ]


def filter_triangle_combinations(points, max_distance):
    """Get combinations of points forming triangles within a certain distance."""
    return [
        (c1, c2, c3)
        for c1, c2, c3 in get_combinations(range(len(points)), 3)
        if all(
            distance(points[i], points[j]) < max_distance
            for i, j in get_combinations([c1, c2, c3], 2)
        )
    ]


def circles_intersect(p1, p2, R):
    return distance(p1, p2) < 2 * R


def filter_combinations_by_circle_intersection(points, R, comb_size):
    filtered_combinations = []
    for comb in get_combinations(range(len(points)), comb_size):
        if all(
            circles_intersect(points[i], points[j], R)
            for i, j in get_combinations(comb, 2)
        ):
            filtered_combinations.append(comb)
    return filtered_combinations


def all_circles_intersect(points, R):
    center_of_mass = np.mean(points, axis=0)
    return all(distance(center_of_mass, p) < R for p in points)


def filter_combinations_by_circle_intersection(points, R, comb_size):
    filtered_combinations = []
    for comb in combinations(range(len(points)), comb_size):
        if all_circles_intersect([points[i] for i in comb], R):
            filtered_combinations.append(comb)
    return filtered_combinations


def main():
    st.title("Simplicial Complex Visualization")

    R = st.sidebar.slider("Filtration parameter", 0.1, 5.0, 0.5)
    mode = st.sidebar.selectbox("Simplicial Complex", ["Vietoris-Rips", "Čech"])

    # Generate a default set of points that lie roughly on the vertices of a cube with some noise
    default_points = np.array(
        [
            [0, 0, 0],
            [0.5, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.5, 0.5, 0],
            [1, 0, 1],
            [0, 1, 0.5],
            [1, 0.5, 1],
        ],
        dtype=np.float64,
    )
    default_points_list = default_points.tolist()
    default_points_str = str(default_points_list)
    point_list_str = st.sidebar.text_area(
        "Enter your list of points", default_points_str
    )

    try:
        centers = np.array(ast.literal_eval(point_list_str))
        if centers.shape[1] != 3:
            st.sidebar.warning("Each point should have exactly 3 coordinates.")
            return
    except (ValueError, SyntaxError):
        st.sidebar.warning("Invalid input. Make sure to enter a valid list of points.")
        return

    draw_balls = st.sidebar.checkbox("Draw Balls around Points", False)

    # Initialize figure
    fig = go.Figure()

    # Generate scatter plot for sphere centers
    fig.add_trace(
        go.Scatter3d(x=centers[:, 0], y=centers[:, 1], z=centers[:, 2], mode="markers")
    )

    # Filter combinations for edges or circles based on the selected mode
    if mode == "Vietoris-Rips":
        edge_combinations = filter_combinations_by_distance(centers, 2 * R)
    else:
        edge_combinations = filter_combinations_by_circle_intersection(centers, R, 2)

    edge_x, edge_y, edge_z = [], [], []
    for c1, c2 in edge_combinations:
        p1, p2 = centers[c1], centers[c2]
        edge_x.extend([p1[0], p2[0], None])
        edge_y.extend([p1[1], p2[1], None])
        edge_z.extend([p1[2], p2[2], None])

    # Add edges to figure
    fig.add_trace(
        go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z, mode="lines", line=dict(color="blue", width=2)
        )
    )

    # Calculate and draw triangles and tetrahedra
    if mode == "Vietoris-Rips":
        tetrahedra_combinations = filter_tetrahedra_combinations(centers, 2 * R)
        triangle_combinations = filter_triangle_combinations(centers, 2 * R)
    else:
        tetrahedra_combinations = filter_combinations_by_circle_intersection(
            centers, R, 4
        )
        triangle_combinations = filter_combinations_by_circle_intersection(
            centers, R, 3
        )

    tet_i, tet_j, tet_k = [], [], []
    for c1, c2, c3, c4 in tetrahedra_combinations:
        tet_i.extend([c1, c2, c1, c2, c1, c2])
        tet_j.extend([c2, c3, c3, c4, c4, c3])
        tet_k.extend([c3, c4, c4, c1, c2, c1])

    tri_i, tri_j, tri_k = [], [], []
    for c1, c2, c3 in triangle_combinations:
        tri_i.append(c1)
        tri_j.append(c2)
        tri_k.append(c3)

    # Add shaded triangles
    if tri_i:
        fig.add_trace(
            go.Mesh3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                i=tri_i,
                j=tri_j,
                k=tri_k,
                opacity=0.5,
                color="#FFB6C1",
            )
        )

    # Add shaded tetrahedra
    if tet_i:
        fig.add_trace(
            go.Mesh3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                i=tet_i,
                j=tet_j,
                k=tet_k,
                opacity=0.6,
                color="#FF69B4",
            )
        )

    if draw_balls:
        draw_spheres(fig, centers, R)  # Step 2: Draw spheres if option is enabled

    # Update layout and show the figure in Streamlit
    fig.update_layout(
        width=800, height=800, scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    )
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()
