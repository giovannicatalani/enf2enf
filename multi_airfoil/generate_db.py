import numpy as np
import matplotlib.pyplot as plt
from sdf_utils import MultiplePolygons, Polygon  # Your existing SDF utilities
import numpy as np
import matplotlib.pyplot as plt

class MultiElementAirfoilSDF:
    """
    MultiElementAirfoilSDF loads a multielement airfoil from a file,
    defines separate polygons for the main element and the flap,
    The input file is assumed to have (approximately) four columns:
        main_x, main_y, flap_x, flap_y
    Some lines may have only two columns (main element only). In that case,
    the flap coordinates are set to NaN.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.main_vertices = None  # (N,2) for main element
        self.flap_vertices = None  # (M,2) for flap element

    def load_and_reorder_data(self):
        # Use a manual loader that reads each line and pads missing values.
        data_list = []
        with open(self.filepath, 'r') as f:
            for line in f:
                # Strip and skip empty or comment lines.
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                tokens = line.split()
                # If only two tokens, assume flap data is missing.
                if len(tokens) == 2:
                    tokens += [np.nan, np.nan]
                # Otherwise, if there are four tokens, keep them.
                if len(tokens) < 4:
                    # Skip lines that don't have at least two numbers.
                    continue
                try:
                    row = [float(tok) for tok in tokens[:4]]
                    data_list.append(row)
                except Exception as e:
                    # Skip lines that cause errors.
                    continue
        
        if not data_list:
            raise ValueError("No valid data was found in the file.")
        data = np.array(data_list)
        # Expect columns: [main_x, main_y, flap_x, flap_y]
        # Extract main element coordinates (assume always present)
        main = data[:, 0:2]
        # Extract flap coordinates: select only rows where flap_x is not NaN.
        flap_mask = ~np.isnan(data[:, 2])
        flap = data[flap_mask, 2:4]
        
        # Reorder each polygon using the trailing edge index strategy.
        self.main_vertices = self._reorder_polygon(main)
        if flap.shape[0] > 0:
            self.flap_vertices = self._reorder_polygon(flap)
        else:
            self.flap_vertices = None

    def _reorder_polygon(self, pts):
        """
        Reorder a set of 2D points to form a closed polygon.
        Assumes the upper surface is from leading edge to trailing edge.
        Splits the points at the index where x is maximum.
        """
        # Find the trailing edge (largest x coordinate)
        trailing_edge_index = np.argmax(pts[:, 0])
        upper = pts[:trailing_edge_index+1]
        lower = pts[trailing_edge_index+1:][::-1]
        poly = np.vstack((upper, lower))
        # Remove duplicate endpoint if present.
        if np.linalg.norm(poly[0] - poly[-1]) < 1e-6:
            poly = poly[:-1]
        return poly
    


def transform_flap(vertices, angle_deg, translation):
    """
    Apply rotation and translation to flap vertices.
    
    Args:
        vertices: np.array of shape (N,2) containing flap vertices
        angle_deg: rotation angle in degrees
        translation: [dx, dy] translation vector
    """
    # Convert angle to radians
    angle = np.deg2rad(angle_deg)
    
    # Create rotation matrix
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    
    # Apply rotation and translation
    transformed = (vertices @ R.T) + translation
    return transformed


def check_overlap(main_poly, flap_vertices):
    """
    Check if transformed flap overlaps with main element by computing
    the SDF of the main element at each flap vertex.
    Returns True if there is overlap.
    """
    # Compute SDF values for the main element at each flap vertex
    sdf_values = main_poly.sdf(flap_vertices)
    
    # If any vertex has negative SDF, there's overlap
    return np.any(sdf_values < 0)

def generate_variant_dataset(main_vertices, flap_vertices, num_variants=200, num_points=10000):
    """
    Generate variants of multi-element airfoil by transforming the flap.
    
    Args:
        main_vertices: vertices of main airfoil element
        flap_vertices: vertices of flap element
        num_variants: number of variants to generate
        num_points: number of points to sample for SDF computation
    
    Returns:
        Dictionary containing variants with their SDFs
    """
    # Create polygon object for the main element
    main_poly = Polygon(main_vertices)
    variants = {}
    count = 0
    attempts = 0
    max_attempts = 10000
    
    # Parameters for variant generation
    angle_range = (-5, 5)  # degrees
    translation_range = np.array([[-0.05, -0.05], [0.01, 0.01]])  # min/max for x,y
    
    while count < num_variants and attempts < max_attempts:
        attempts += 1
        
        # Generate random transformation
        angle = np.random.uniform(*angle_range)
        translation = np.random.uniform(translation_range[0], translation_range[1])
        
        # Transform flap
        transformed_flap = transform_flap(flap_vertices, angle, translation)
        
        # Check for overlap
        if check_overlap(main_poly, transformed_flap):
            continue
            
        # Create multi-polygon object for SDF computation
        multi_poly = MultiplePolygons([main_vertices, transformed_flap])
        
        # Sample points and compute SDF
        points, sdf = multi_poly.sample_points(num_points=num_points)
        
        # Store variant data
        variants[count] = {
            'angle_deg': angle,
            'translation': translation,
            'flap_variant': transformed_flap,
            'points': points,
            'sdf': sdf
        }
        
        print(f"Generated variant {count}: angle={angle:.2f}°, translation={translation}")
        count += 1
    
    print(f"Generated {count} variants after {attempts} attempts")
    return variants

def visualize_variant(main_vertices, flap_vertices, points, sdf, title=""):
    """
    Visualize a variant with its SDF field.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scatter plot
    scatter = ax1.scatter(points[:, 0], points[:, 1], c=sdf, 
                         cmap='viridis', s=1, alpha=0.5)
    ax1.plot(main_vertices[:, 0], main_vertices[:, 1], 'k-', 
             linewidth=2, label='Main')
    ax1.plot(flap_vertices[:, 0], flap_vertices[:, 1], 'r-', 
             linewidth=2, label='Flap')
    plt.colorbar(scatter, ax=ax1)
    ax1.set_title(f'SDF Field {title}')
    ax1.legend()
    
    # Contour plot
    contour = ax2.tricontour(points[:, 0], points[:, 1], sdf, 
                            levels=20, cmap='viridis')
    ax2.plot(main_vertices[:, 0], main_vertices[:, 1], 'k-', linewidth=2)
    ax2.plot(flap_vertices[:, 0], flap_vertices[:, 1], 'r-', linewidth=2)
    plt.colorbar(contour, ax=ax2)
    ax2.set_title(f'SDF Contours {title}')
    
    # Save the figure
    plt.savefig('variant_visualization.png')
    plt.close() 

if __name__ == "__main__":
    # Load airfoil data (using your provided coordinates)
    file_path = 'fowler_flap_airfoil.txt'
    multi_airfoil = MultiElementAirfoilSDF(file_path)
    multi_airfoil.load_and_reorder_data()
    main_vertices = multi_airfoil.main_vertices
    flap_vertices = multi_airfoil.flap_vertices
    
    # Generate variant dataset
    variants = generate_variant_dataset(main_vertices, flap_vertices, 
                                     num_variants=10, num_points=10000)
    
    # Visualize original configuration
    multi_poly = MultiplePolygons([main_vertices, flap_vertices])
    points, sdf = multi_poly.sample_points(num_points=10000)
    visualize_variant(main_vertices, flap_vertices, points, sdf, 
                     title="(Original)")
    
    # Visualize some random variants
    for idx in np.random.choice(list(variants.keys()), 1, replace=False):
        var = variants[idx]
        visualize_variant(
            main_vertices, 
            var['flap_variant'],
            var['points'], 
            var['sdf'],
            title=f"(Variant {idx}: {var['angle_deg']:.1f}°, {var['translation']})"
        )
    
    np.save("geom_db.npy", variants)