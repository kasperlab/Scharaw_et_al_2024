# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:49:56 2024

@author: Karl Annusver
"""

import math
import scipy
import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import skimage
from skimage.measure import label, regionprops

from IPython.display import clear_output

def initialize_subplots(groups_to_plot, ncols = 3, figsize_multiplier = (7,5), gridspec_kw = None, figsize = None, print_help = True, **fig_kw):
    if type(groups_to_plot)==list:
        total = len(groups_to_plot)
    else:
        total = groups_to_plot
    nrows = int(np.ceil(total/ncols))
    if not figsize:
        figsize = (figsize_multiplier[0]*ncols, figsize_multiplier[1]*nrows)
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize, gridspec_kw = gridspec_kw, **fig_kw)
    if print_help:
        if nrows>1 and ncols>1:
            print('ax = axes[ix // ncols, ix % ncols]')
        else:
            print('ax = axes[ix]')
    return fig, axes

def get_ax(axes, ix, ncols):
    """
    Helper function to return the correct axis
    """
    if len(axes.shape) > 1:
        return axes[ix // ncols, ix % ncols]
    else:
        return axes[ix]

def get_annotation_coordinates(image_frames):
    # Initialize the dataframe
    columns = ['ImageNumber', 'ObjectNumber', 'Location_Center_X', 'Location_Center_Y']
    df = pd.DataFrame(columns=columns)
    
    rows = []
    # Process each frame
    for i, frame in enumerate(image_frames):
        labeled_frame = label(frame)
        props = regionprops(labeled_frame)
  
        for j, prop in enumerate(props):
            y, x = prop.centroid
            rows.append({
                'ImageNumber': i + 1,
                'ObjectNumber': j + 1,
                'Location_Center_X': x,
                'Location_Center_Y': y
            })
    return pd.concat([df, pd.DataFrame(rows)], ignore_index = True)


#######################################################################
### Order points based on their location and assign them to ROIs ######
#######################################################################

def cartesian_to_polar(coords, center, ref_line):
    """ Convert Cartesian coordinates to polar (radius and angle) relative to a center and a reference line. """
    rel_coords = coords - center  # Shift coordinates relative to the center
    radius = np.sqrt(np.sum(rel_coords**2, axis=1))  # Calculate radius for each point

    # Calculate the angle of the reference line
    ref_line_vector = np.array(ref_line[1]) - np.array(ref_line[0])
    ref_line_angle = np.arctan2(ref_line_vector[1], ref_line_vector[0])
    
    # Calculate angle of each point and adjust by the reference line angle
    angle = np.arctan2(rel_coords[:, 1], rel_coords[:, 0]) - ref_line_angle
    # Normalize the angle to be between -pi and pi
    angle = np.mod(angle + np.pi, 2*np.pi) - np.pi

    return radius, angle

def sort_by_angle(coords, center, ref_line):
    """ Sort coordinates in each set by angle. Vectorized version. """
    _, angles = cartesian_to_polar(coords, center, ref_line)  # Convert to polar coordinates
    return coords[np.argsort(angles)]  # Sort by angle

def group_coordinates(sets, ref_line):
    """ Group coordinates from different sets based on their order after sorting by angle. """
    center = np.mean(sets[0], axis=0)  # Compute the center from the first set
    ref_line = [ref_line, center]  # Define the reference line
    # Sort each set by angle and stack them together
    sorted_sets = [sort_by_angle(np.array(set), center, ref_line) for set in sets]
    return np.stack(sorted_sets, axis=-1), center

def get_sorted_indeces(grouped_coords, input_coords, ix):
    """ Find indexes where the input coordinate matches the grouped and sorted coordinates. """
    # Find matching indexes for the given input coordinates
    return np.where(np.all(grouped_coords[:, :, ix] == input_coords[:, None], axis=-1))[1]

def plot_polar_sorted_plotter(coords, center, ref_line, ax, color, s=3):
    """Plots sorted and original points on a polar coordinate system."""
    # Convert to polar coordinates
    radius, angles = cartesian_to_polar(coords, center, [ref_line, center])
    # Sort coordinates by angles
    sorted_coords = sort_by_angle(coords, center, [ref_line, center])

    # Plot the paths and points
    ax.plot(sorted_coords[:, 0], sorted_coords[:, 1], label='Sorted Path', color='lightgray', zorder=-1)
    ax.scatter(coords[:, 0], coords[:, 1], label='Original Points', color=color, s=s, zorder=1)

def plot_polar_sorted(coords, center, ref_line, ax, colors, s=3, frame=0):
    """Plots multiple sets of coordinates after sorting them based on polar conversion."""
    for i, coord_set in enumerate(coords):
        plot_polar_sorted_plotter(coord_set, center, ref_line, ax, color=colors[i], s=s)
    ax.scatter(center[0], center[1], color='red', label='Center', s=s)
    ax.set_title(f'Frame: {frame}', y=0.88)

def apply_rotation(row, column_name, rad, adjustment_rad):
    """Apply rotation to a point around a given origin."""
    return rotate(row['Lumen'], row[column_name], rad - row['Angle'] + adjustment_rad)

def rotate(origin, point, rad):
    """Rotate a point around an origin by a specified angle in radians."""
    ox, oy = origin
    px, py = point
    
    qx = ox + math.cos(rad) * (px - ox) - math.sin(rad) * (py - oy)
    qy = oy + math.sin(rad) * (px - ox) + math.cos(rad) * (py - oy)
    
    return qx, qy


#Function to smooth the ROIs for nicer presentation
def smooth_polygon(x,y, skip_points = 10):
    x = x[::skip_points]
    y = y[::skip_points]
    
    orig_len = len(x)
    x = x[-3:-1] + x + x[1:3]
    y = y[-3:-1] + y + y[1:3]
    t = np.arange(len(x))
    ti = np.linspace(2, orig_len + 1, 10 * orig_len)

    xi = scipy.interpolate.interp1d(t, x, kind='cubic')(ti)
    yi = scipy.interpolate.interp1d(t, y, kind='cubic')(ti)
    
    return xi, yi


###############################
## Calculate bisecting lines ##
###############################

def find_bisecting_lines_and_distances(row, include_bm_lumen_points_in_line = False, outside_shift = 10, quantile_percent = 10):
    """
    Process a single row of the DataFrame.
    """
    if include_bm_lumen_points_in_line:
        x, y = include_bm_lumen_points(row, outside_shift = outside_shift)
    else:
        x, y = zip(*row['Roi_coords_smooth'])
    x, y, shift_x, shift_y = shift_coordinates(x, y)
    rr, cc, mean_cc, unique_rr = calculate_bisecting_line(x, y)
    
    # Calculate the distance of the mean line from the corresponding y-axis border of the polygon.
    distance_to_border = abs(mean_cc - np.array([cc[rr == y_val].max() for y_val in unique_rr]))
    
    #Return values to original coordinates
    mean_cc -= shift_x
    unique_rr = unique_rr.astype(np.float64) - shift_y
    
    return pd.Series({
        'bisecting_line': list(zip(*adjust_bisect_line_extremes(mean_cc, unique_rr, quantile_percent = 10))),
        'distance_to_border': distance_to_border
    })

def adjust_bisect_line_extremes(mean_cc, unique_rr, quantile_percent = 10):
    """
    Adjusts the extremes of a bisect line based on quantile thresholds.
    """
    # Convert input tuples to numpy arrays for efficient element-wise operations
    x, y = np.array(mean_cc), np.array(unique_rr)

    # Iterate through the quantile thresholds to adjust the line extremes
    for i in np.quantile(y, [quantile_percent / 100, 1 - (quantile_percent / 100)]):
        # Find the x-value closest to the current quantile of y
        closest_x = x[np.argmin(abs(y - i))]

        # Update x-values where y is greater/less than the current quantile
        # The condition 'i > 0' checks if we are dealing with the upper quantile
        x = np.where((y > i) if i > 0 else (y < i), closest_x, x)

    return x, y


def include_bm_lumen_points(row, outside_shift = 5):
    """Include lumen and BM marked coordinates in the ROI coordinate list for better bisecting line calculation"""
    def get_side_of_shape(y_coords, reference_y):
        """ Determine the side of the shape based on the reference y-coordinate """
        return 'bottom' if reference_y < np.mean(y_coords) else 'top'
    
    def adjust_coordinates_if_inside(lumen_bm_coords, patch, y_coords, sides, outside_shift = 5):
        """ Adjust coordinates if they are inside the ROI """
        #contained = patch.contains_points(lumen_bm_coords)
        #if not any(contained):
        #    return lumen_bm_coords
    
        adjusted_coords = lumen_bm_coords.copy()
        #for idx, is_contained in enumerate(contained):
        for idx, coord in enumerate(lumen_bm_coords):
            #if is_contained:
            dot_type = 'lumen' if idx == 0 else 'bm'
            dot_side = sides[dot_type]
            y_limit = np.min(y_coords) if dot_side == 'bottom' else np.max(y_coords)
            adjusted_coords[idx, 1] = y_limit - outside_shift if dot_side == 'bottom' else y_limit + outside_shift
        return adjusted_coords
    
    x, y = zip(*row['Roi_coords_smooth'])
    (l_x, l_y), (b_x, b_y) = row[['Lumen', 'BM']]
    x, y = np.array(x), np.array(y)
    
    side = get_side_of_shape(y, l_y)
    sides = {'lumen': side, 'bm': 'top' if side == 'bottom' else 'bottom'}
    
    # Adjust lumen and bm coordinates if inside the ROI
    patch = row['Patches']
    lumen_bm_points = np.array([[l_x, l_y], [b_x, b_y]])
    lumen_bm_points = adjust_coordinates_if_inside(lumen_bm_points, patch, y, sides, outside_shift)
    l_x, l_y = lumen_bm_points[0]
    b_x, b_y = lumen_bm_points[1]
    
    # Integrate the adjusted coordinates into the shape
    if sides['lumen'] == 'bottom':
        x = np.concatenate([[b_x], x, [l_x]])
        y = np.concatenate([[b_y], y, [l_y]])
    else:
        x = np.concatenate([[l_x], x, [b_x]])
        y = np.concatenate([[l_y], y, [b_y]])
    return x, y


def shift_coordinates(x, y):
    """
    Shifts coordinates to ensure all are positive.
    """
    #x, y = zip(*coords)
    x, y = np.array(x), np.array(y)
    shift_x = abs(min(x))
    shift_y = abs(min(y))
    x += shift_x
    y += shift_y
    return x, y, shift_x, shift_y

def calculate_bisecting_line(x, y):
    """
    Calculates the mean x coordinate for each unique y value of the polygon.
    r - row, c - column
    """
    rr, cc = skimage.draw.polygon(y, x)
    unique_rr, inverse_indices = np.unique(rr, return_inverse=True)
    mean_cc = np.bincount(inverse_indices, weights=cc) / np.bincount(inverse_indices)
    return rr, cc, mean_cc, unique_rr


#########################
## Calculate distances ##
#########################

def interpolate_coordinates(y_coords, x_coords, new_y):
    """
    Interpolate x-coordinates corresponding to new y-coordinates.
    """
    f = scipy.interpolate.interp1d(y_coords, x_coords, kind='linear', fill_value='extrapolate')
    return f(new_y)


def calculate_side_and_roi_distances(center_norm, roi_name = 'Roi_coords'):
    """
    Calculate distances and sides relative to the bisecting line and ROIs for each cell frame.
    
    Parameters:
    - center_norm: DataFrame containing normalized centers and other geometric data per frame.
    - roi_name: String, the column name for ROI coordinates within the DataFrame. Defaults to 'Roi_coords'.

    Returns:
    - Tuple of three lists containing distances from the bisecting line, distances from the ROI, and side information.
    """
    # Initialize lists to store results for sides, bisecting line distances, and ROI distances.
    res_sides, res_bisect_distances, res_roi_distances = [], [], []
    
    # Iterate over each group of data in the DataFrame
    for (frame, group), row in center_norm.iterrows():
        # Unpack coordinate tuples for each type of line in the DataFrame.
        x, y = zip(*row['coordinates_in_roi'])
        bisect_x, bisect_y = zip(*row['bisecting_line'])
        roi_x, roi_y = zip(*row[roi_name])
        
        # Convert coordinate lists to numpy arrays for easier mathematical operations.
        x, y = np.array(x), np.array(y)
        bisect_x, bisect_y = np.array(bisect_x), np.array(bisect_y)
        roi_x, roi_y = np.array(roi_x), np.array(roi_y)
    
        # Interpolate to find x-coordinates on the bisecting line closest to y-coordinates of cells.
        closest_bisect_x = np.array(interpolate_coordinates(bisect_y, bisect_x, y))
        # Determine which side of the bisecting line each coordinate lies.
        sides = np.where(x < closest_bisect_x, -1, 1)
        
        # Calculate perpendicular distances to the bisecting line.
        closest_x = np.array(interpolate_coordinates(bisect_y, bisect_x, y))
        bisect_distances = np.abs(closest_x - x)
        
        # Calculate x-coordinates on the bisecting line that correspond to the y-coordinates of ROIs.
        closest_roi_x = np.array(interpolate_coordinates(bisect_y, bisect_x, roi_y))
        # Dictionary to store indices of ROI coordinates based on their sides.
        roi_side_dict = {-1: np.where(roi_x < closest_roi_x)[0], 1: np.where(roi_x >= closest_roi_x)[0]}
        
        # Initialize an array to store distances to the ROI.
        roi_distances = np.copy(x)
        for side in [-1, 1]:
            ixs = roi_side_dict[side]
            coord_ixs = np.where(sides == side)[0]
            # Calculate distances from each coordinate to the closest point on the ROI on the same side.
            roi_distances[coord_ixs] = abs(roi_distances[coord_ixs] - np.array(interpolate_coordinates(roi_y[ixs], roi_x[ixs], np.array(y)[coord_ixs])))
            
        # Append results for this group to the respective lists.
        res_bisect_distances.append(bisect_distances)
        res_roi_distances.append(roi_distances)
        res_sides.append(sides)
    return res_bisect_distances, res_roi_distances, res_sides

def calculate_distance_fraction(distance_to_bisect, distance_to_roi):
    total = np.sum(list(zip(*[distance_to_bisect, distance_to_roi])), axis = 1)
    return distance_to_bisect/total




##############
## Plotting ##
##############

def find_shift_for_patches(series, additional_shift = 5):
    """
    Calculate the cumulative shift for each patch in a series, with an additional shift applied between patches.
    """
    shift = [0]
    extents = series.apply(lambda x: (min(np.array(x)[:, 0]), max(np.array(x)[:, 0])))
    differences = [abs(prev[0]) + curr[1] + additional_shift for prev, curr in zip(extents.shift(-1), extents) if prev is not None]
    shift.extend(differences)
    return np.cumsum(shift)
    
def shift_x_coordinates_for_side_by_side(row):
    """
    Shift x-coordinates of a row's ROI coordinates by the specified shift amount for side-by-side arrangement.
    """
    coords, shift = row[['Roi_coords', 'roi_shift']]
    x, y = zip(*coords)
    x += np.array(x) + shift
    return list(zip(x, y))


# Make histograms per group and frame
def find_max_intensity_per_group(df, group_column, n_bins = 30, intensity_cutoff = 1, values_to_plot = 'distance_fraction_bisect_to_roi'):
    # Pre-calculate the maximum histogram value for each group to normalize y-axis later.
    max_values = {}
    df = df.reset_index()
    groups = df[group_column].values
    for g in set(groups):
        # Extract relevant data for the current group.
        f, i, side = zip(*df.query(f'{group_column} == @g')[[values_to_plot, 'intensities', 'side_of_cell']].values)
        if intensity_cutoff == None:
            pass
        elif values_to_plot != 'distance_fractions_by_intensity':
            f = [np.nan_to_num(val)*side[ix] for ix, val in enumerate(f)]
            f = [val[i[ix]>intensity_cutoff] for ix, val in enumerate(f)]
            
        # Find the maximum histogram count for each group.
        max_values[g] = np.max([np.max(np.histogram(values, bins = n_bins, range = (-1, 1))[0]) for values in f])
    return max_values

def plot_histograms_per_time_and_group(center_norm, axes, intensity_cutoff = 1, n_bins = 30, max_values = None, frame_col = 'ImageNumber', palette = 'RdYlGn', weights = None, stat = 'count', values_to_plot = 'distance_fraction_bisect_to_roi'):
    frames = center_norm.reset_index()[frame_col].unique()
    
    # Plot histograms for each frame and group.
    for (frame, group), row in center_norm.iterrows():
        print(frame, group)
        clear_output(wait = True)
        
        ax = axes[list(frames).index(frame), group] #quick fix because we don't have 30 frames
        if intensity_cutoff != None:
            # Extract and process the intensity, fraction distance, and side data.
            i, f, side = row[['intensities', values_to_plot, 'side_of_cell']]
            f = np.array(f) * side
            f = f[i > intensity_cutoff]
        else:
            f, side = row[[values_to_plot, 'side_of_cell']]
            #f = np.array(f)*side
            
        if weights == 'intensity':
            weight_values = i[i > intensity_cutoff]
        else:
            weight_values = None
            
        # Create histogram plot.
        sbn.histplot(x = f, bins = n_bins, kde = False, ax = ax, stat = stat, weights = weight_values)
        # Customize the subplot.
        ax.set_xlabel('') 
        if frame == frames[0]:
            ax.set_title(group)
        if frame != frames[-1]:
            ax.set_xticklabels([])
        ax.set_ylabel(frame if group == 0 else '')
        ax.set_yticks([])
        if isinstance(max_values, dict):
            ax.set_ylim(top = max_values[group]+5)
        ax.set_xlim([-1, 1])
        if palette != None:
            # Set up colormap and normalization for coloring bars in the histogram.
            #cmap = plt.cm.RdYlGn if group % 2 == 0 else plt.cm.RdYlGn_r
            cmap = sbn.color_palette(palette if group % 2 == 0 else f'{palette}_r', as_cmap = True)
            norm = mpl.colors.Normalize(vmin=-1, vmax=1)
            for bar in ax.patches:
                # Get the center of each bar
                bar_center = (bar.get_x() + bar.get_width() / 2)
                # Normalize the center value and get the corresponding color
                color = cmap(norm(bar_center))
                # Set the color for the bar
                bar.set_facecolor(color)
        [ax.spines[spine].set_visible(False) for spine in ['top', 'right']]
        

def plot_rois_on_right_side_axis(center_df, axes, group_col = 'Ordered_Number', frame_col = 'ImageNumber', coord_name = 'Roi_coords', colorize = True, cmap = 'tab10'):
    frames = center_df.reset_index()[frame_col].unique()
    # Plot the ROIs on the last column of subplots for each frame.
    for ix, frame in enumerate(set(frames)):
        ax = axes[ix, -1]
        df = center_df.loc[frame].reset_index()
        groups = center_df.reset_index()[group_col].unique()
        for i, row in df.iterrows():
            x, y = row['BM']
            g = row[group_col]
            patch = mpl.patches.Polygon(row[coord_name], closed = True, fill = False, alpha = 0.5, ec = sbn.color_palette(cmap)[i] if colorize else None)
            ax.add_patch(patch)
            ax.autoscale()
            if g in [min(groups), max(groups)]:
                ax.text(x = x, y = y, s = g)
    
        # Customize the subplot appearance.
        [ax.spines[spine].set_visible(False) for spine in ax.spines]
        ax.set_yticks([])
        ax.set_xticks([])
        ax.autoscale()

def calc_density(x: np.ndarray, y: np.ndarray):
    """\
    Function to calculate the density of cells in an embedding.
    Taken from scanpy embedding density calculation
    """
    from scipy.stats import gaussian_kde

    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    min_z = np.min(z)
    max_z = np.max(z)

    # Scale between 0 and 1
    scaled_z = (z - min_z) / (max_z - min_z)

    return scaled_z


#######################
## Combined analysis ##
#######################

def min_max_scale_roi_y_coordinates(roi_coords):
    x, y = zip(*roi_coords)
    return 2 * (y-np.min(y))/(np.max(y) - np.min(y)) - 1


def calc_density_1d(x: np.ndarray):
    """\
    Function to calculate the density of cells in a 1D array.
    Adapted from scanpy embedding density calculation.
    """
    # Calculate the point density
    z = gaussian_kde(x)(x)

    min_z = np.min(z)
    max_z = np.max(z)

    # Scale between 0 and 1
    scaled_z = (z - min_z) / (max_z - min_z)

    return scaled_z