import copy
import osmnx as ox
import momepy
import pandas as pd
import geopandas as gpd
import gemgis as gg

from shapely.geometry import MultiPoint, LineString
from shapely.ops import polygonize, unary_union, nearest_points, voronoi_diagram
from shapely.strtree import STRtree

from utilities import relation2dict, check_crs
from geometry import clean_geometry, poly2line, close_holes, dissolve_poly


def morphological_tessellation(element, size=10, buffer=100):
    """
    Create a Voronoi Tessellation based on the geometries from a GeoDataFrame.

    Parameters
    ----------
    element : geopandas.GeoDataFrame
        the GeoDataFrame containing all the geometry
    size : float (default 10)
        the elements whose area below this number will be regarded as additional structure,
        and be subjected to elimination by a list of criteria
    buffer : float (default 100)
        the catchment area of each element
        this parameter sets the maximal distance from elements as the tessellation geometry

    Returns
    -------
    tessellation_gdf : geopandas.GeoDataFrame
        the GeoDataFrame containing the tessellation result
    """
    # The Momepy only works for projected CRS
    projected = ox.project_gdf(element)

    # Conduct a preprocessing step
    projected = momepy.preprocess(projected.reset_index(), size,
                                     compactness=0.2, islands=True, verbose=False)
    # Generate unique ID for later use
    projected['uID'] = momepy.unique_id(projected)
    # Generate the limit of tessellation using the buffer method
    limit = momepy.buffered_limit(projected, buffer)

    # Conduct tessellation
    tessellation = momepy.Tessellation(projected, unique_id='uID', limit=limit)

    # Convert to GeoDataFrame
    tessellation_gdf = tessellation.tessellation
    # Convert back the CRS
    tessellation_gdf = tessellation_gdf.to_crs(crs=element.crs)

    return tessellation_gdf


def voronoi_tessellation(element, boundary=None, tolerance=0.0):
    """
    Create a Voronoi Tessellation based on the geometries from a GeoDataFrame.

    Parameters
    ----------
    element : geopandas.GeoDataFrame
        the GeoDataFrame containing all the geometry
    boundary : shapely.geometry.Polygon
        the envelope used to clip the resulting diagram
    tolerance : float (default 0.0)
        snap nearby points that are within the specified tolerance distance to a common point
        a tolerance of 0.0 specifies that no snapping will take place

    Returns
    -------
    tessellation_gdf : geopandas.GeoDataFrame
        the GeoDataFrame containing the tessellation result
    """
    # The calculation is suggested to happen on a projected CRS
    element_projected = ox.project_gdf(element)

    # Convert the original element to centroid in a GeoDataFrame
    centroids = element_projected.geometry.centroid
    # Create a MultiPoint geometry
    multi_point = MultiPoint(list(centroids))
    # Calculate the voronoi diagram
    voronoi = voronoi_diagram(multi_point, envelope=boundary, tolerance=tolerance)
    # Get the result in a GeoDataFrame
    tessellation_gdf = gpd.GeoDataFrame(geometry=list(voronoi.geoms), crs=element_projected.crs)

    # Convert back the CRS
    tessellation_gdf = tessellation_gdf.to_crs(crs=element.crs)

    return tessellation_gdf


def slice_tessellation(tessellation, region, element):
    """
    Slice the tessellation polygons within the frame of corresponding spatial units
    as a further way of partitioning.

    Parameters
    ----------
    tessellation : geopandas.GeoDataFrame
        the GeoDataFrame containing the tessellation polygons
    region : geopandas.GeoDataFrame
        the GeoDataFrame containing the spatial unit
    element : geopandas.GeoDataFrame
        the GeoDataFrame containing the elements (such as buildings)

    Returns
    -------
    filtered_outcome : geopandas.GeoDataFrame
        the GeoDataFrame with the tessellation polygons been sliced
    """
    # Check the CRS
    df_lst = [tessellation, region, element]
    if check_crs(df_lst) == False:
        return

    tessellation = clean_geometry(tessellation)
    region = clean_geometry(region)
    element = clean_geometry(element)

    # ----- Slice the tessellation result -----
    region_relation, _ = relation2dict(region, tessellation, predicate="intersects")

    sliced_tessellation = []

    for key, value in region_relation.items():
        if not len(value) == 0:
            for i in value:
                # We only keep those intersected geometries
                intersection = region.geometry[key].intersection(tessellation.geometry[i])
                sliced_tessellation.append(intersection)

    outcome = gpd.GeoDataFrame(geometry=sliced_tessellation, crs=region.crs)

    # Remove empty geometry and explode multipart geometries
    outcome = outcome[~outcome.is_empty]
    outcome = outcome.explode(ignore_index=True)
    outcome.reset_index(drop=True, inplace=True)

    # ----- Remove the sliced result where no element situated within -----
    outcome_relation, _ = relation2dict(outcome, element, predicate="intersects")

    removal = []
    for key, value in outcome_relation.items():
        if len(value) == 0:
            removal.append(key)

    filtered_outcome = outcome.drop(list(set(removal)))

    return filtered_outcome


def fill_tessellation_gap(tessellation, region):
    """
    Fill in the gap of each spatial unit if it has not been fully covered by the tessellation polygons.

    Parameters
    ----------
    tessellation : geopandas.GeoDataFrame
        the GeoDataFrame containing the tessellation polygons
    region : geopandas.GeoDataFrame
        the GeoDataFrame containing the spatial unit

    Returns
    -------
    division_gdf : geopandas.GeoDataFrame
        the GeoDataFrame with the tessellation result with all the gaps been filled
    """
    # Check the CRS
    df_lst = [tessellation, region]
    if check_crs(df_lst) == False:
        return

    # Get the relationship into two dicts
    region_tessellation_relation, _ = relation2dict(region, tessellation,
                                                   predicate="intersects")

    gaps = []
    for i in region.index:
        if len(region_tessellation_relation[i]) > 0:
            # Locate to those tessellations within this block
            tessellation_subset = tessellation.loc[region_tessellation_relation[i]]
            # Dissolve these tessellations if they are adjacent to each other
            if len(region_tessellation_relation[i]) > 1:
                tessellation_subset = dissolve_poly(tessellation_subset)
            tessellation_subset.reset_index(drop=True, inplace=True)

            void = region.geometry[i].difference(tessellation_subset.geometry[0])

            if len(tessellation_subset) > 1:
                for idx in range(1, len(tessellation_subset)):
                    void = void.difference(tessellation_subset.geometry[idx])

            if void.geom_type == 'GeometryCollection':
                exploded = gg.vector.explode_geometry_collection(collection=void)
                gaps.extend(exploded)
            else:
                gaps.append(void)

    # Combine two dicts to get the final list of geometries
    division = gaps + list(tessellation.geometry)
    division_gdf = gpd.GeoDataFrame(geometry=division, crs=tessellation.crs)

    return division_gdf


def block_generation(gdf_lst, barrier_idx=None):
    """
    Generate block geometries based on street geometries.

    Parameters
    ----------
    gdf_lst : list[geopandas.GeoDataFrame]
        the list of GeoDataFrames which contain the geometries to create urban block
    barrier_idx : list
        The index of GeoDataFrames within gdf_lst that act as natural or man-made barriers

    Returns
    -------
    block : geopandas.GeoDataFrame
        the GeoDataFrame of street block geometries
    """
    # Check the CRS
    if check_crs(gdf_lst) == False:
        return

    geom_list = []

    # ----- Data Cleaning -----
    for gdf in gdf_lst:
        if 'Polygon' in set(gdf.geom_type) or 'MultiPolygon' in set(gdf.geom_type):
            # Convert Polygons to LineStrings (The returned DataFrame will only include LineStrings)
            gdf = poly2line(gdf)
            geom_list.append(list(gdf.geometry))
        else:
            gdf = clean_geometry(gdf)
            geom_list.append(list(gdf.geometry))

    # Combine all the lists in geom_list
    list_joined = sum(geom_list, [])

    # ----- Block polygon generation -----
    # Obtain the union of street segments
    line_union = unary_union(list_joined)

    lines = []
    for geom in line_union.geoms:
        if geom.geom_type == 'Polygon':
            boundary = geom.boundary
            if boundary.type == 'MultiLineString':
                for line in boundary.geoms:
                    lines.append(line)
            else:
                lines.append(boundary)
        else:
            lines.append(geom)

    # Generate polygons based on segments
    block_geom = list(polygonize(lines))

    # ----- Trim block polygon based on barrier polygons -----
    if barrier_idx:
        # Extract all the barrier polygons to a list
        barrier_poly = []
        for idx in barrier_idx:
            for geom in gdf_lst[idx].geometry:
                barrier_poly.append(geom)

        # Find those block geometries that are equal to the barrier polygons
        removal = []
        tree = STRtree(barrier_poly)

        for i in range(len(block_geom)):
            query = tree.query(block_geom[i])
            for x in query:
                if block_geom[i].equals(x) or block_geom[i].within(x):
                    removal.append(i)

        removal = list(set(removal))

        # Remove these blocks
        removal.sort(reverse=True)

        for idx in removal:
            block_geom.pop(idx)

    # Generate DataFrame
    block = gpd.GeoDataFrame(geometry=block_geom, crs=gdf_lst[0].crs)
    # Close the holes
    block = close_holes(block)

    return block


def snap_block(block, threshold=5):
    """
    Close the geometries that are not closed.

    Parameters
    ----------
    block : geopandas.GeoDataFrame
        the GeoDataFrames of urban block
    threshold : Points within this distance will be snapped into a line

    Returns
    -------
    snapped : geopandas.GeoDataFrame
        the GeoDataFrame with all the geometries been closed
    """
    # Make a deep copy
    block_copy = copy.deepcopy(block)

    new_geom = []
    remove_idx = []

    def segment(line):
        line_segments = list(map(LineString, zip(line.coords[:-1], line.coords[1:])))
        return line_segments

    for i in block.index:
        # Divide the boundary of each block polygon into segments
        bound = segment(block.geometry[i].boundary)
        # Create a DataFrame based on these segments
        bound_df = gpd.GeoDataFrame(geometry=bound, crs=block.crs)

        snap = []
        for j in bound_df.index:
            # for each segment, query those segments that are touched by it
            touch_idx = list(bound_df.sindex.query(bound_df.geometry[j], predicate="touches"))
            # Leave out these segments
            not_touch_idx = [x for x in list(bound_df.index) if x not in touch_idx]
            disjoint = bound_df.loc[not_touch_idx]
            # Find if there is any intersection between it and the buffers of other segments
            disjoint['geometry'] = disjoint['geometry'].buffer(threshold)
            intersection = list(disjoint.sindex.query(bound_df.geometry[j].buffer(threshold), predicate="intersects"))

            if len(intersection) != 0:
                for k in range(0, len(intersection)):
                    # Snap by finding the shortest line
                    pt1, pt2 = nearest_points(bound_df.geometry[j], bound_df.geometry[intersection[k]])
                    shortest_line = LineString([pt1, pt2])
                    snap.append(shortest_line)
            # Drop the line segment we have already calculated
            bound_df.drop([j])

        if len(snap) > 0:
            # Combine all the lists in geom_list
            list_joined = [block.geometry[i].boundary] + snap
            # Obtain the union of street segments
            line_union = unary_union(list_joined)

            lines = []
            if line_union.geom_type != 'GeometryCollection':
                lines.append(line_union)
            else:
                for geom in line_union.geoms:
                    if geom.geom_type == 'Polygon':
                        boundary = geom.boundary
                        if boundary.type == 'MultiLineString':
                            for line in boundary.geoms:
                                lines.append(line)
                        else:
                            lines.append(boundary)
                    else:
                        lines.append(geom)

            # Generate polygons based on these segments
            block_geom = list(polygonize(lines))
            new_geom.extend(block_geom)
            # Noted down the index of blocks to be removed
            remove_idx.append(i)

    # Remove those blocks that will be snapped
    block_copy = block_copy.drop(remove_idx)
    # Create a DataFrame based on the snapped geometries
    snapped_geom = gpd.GeoDataFrame(geometry=new_geom, crs=block.crs)
    # Join the two DataFrames
    snapped = pd.concat([block_copy.geometry, snapped_geom.geometry])

    return snapped
