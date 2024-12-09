### Plotting
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from Utilities_python import genfuel
import Preprocessing

from Preprocessing import fix_bus_numbers

def dfs_for_plotting(gpdf, bus, bus_index, LMP_, lambda_green):
    bus_names = gpdf['Name']
    bus_longs = gpdf['geometry'].x
    bus_lats = gpdf['geometry'].y
    # bus_longs = ercot_coords.point_object.x
    # bus_lats = ercot_coords.point_object.y
    bus_nums = gpdf['Description']
    bus_nums = [s.replace('Bus Number: ', '') for s in bus_nums]
    bus_nums = [int(s.replace('<br/>', '')) for s in bus_nums]
    bus_nums = np.array((bus_nums))


    geometric_points = []
    for xy in zip(bus_longs, bus_lats):
        geometric_points.append(Point(xy))
    
    geo_locations = gpd.GeoDataFrame(gpdf,
                                 crs = {'init': 'epsg:4326'},
                                 geometry = geometric_points)
    
    bus_nums = gpdf['Description']
    bus_nums = [s.replace('Bus Number: ', '') for s in bus_nums]
    bus_nums = [int(s.replace('<br/>', '')) for s in bus_nums]
    gpdf['Description'] = bus_nums

    gpdf = fix_bus_numbers(gpdf, bus[bus_index], bus, 'Description')
    
    # geo_locations['Description'] = bus_nums

    # bs = bus[bus_index]
    # buss = geo_locations['Description']
    # new_bus =[]<
    # for _ , j in enumerate(buss):
    #     for k in range(len(bs)):
    #         if (j==bs[k]):
    #             new_bus.append(bus['bus'][k])
    # gpdf['bus'] = new_bus



    LMP_new = []
    for i in range(len(bus_nums)):
      for j in range(len(bus)):
         if (bus_nums[i] == bus[bus_index][j]):
             LMP_new.append(LMP_[j])
             
    df_map_LMPs = pd.DataFrame()
    df_map_LMPs['bus_lats'] = bus_lats
    df_map_LMPs['bus_longs'] = bus_longs
    df_map_LMPs['LMPs'] = LMP_new
    
    df_green_LMPs = pd.DataFrame()
    df_green_LMPs['bus_lats'] = bus_lats
    df_green_LMPs['bus_longs'] = bus_longs
    df_green_LMPs['LMPs'] = np.array(LMP_new) + lambda_green

    return df_map_LMPs, df_green_LMPs, gpdf

import folium
from folium import plugins
def plot_folium_hm(df_map_LMPs):
    lat_longs = list(map(list, zip(df_map_LMPs["bus_lats"],
                                   df_map_LMPs["bus_longs"],df_map_LMPs["LMPs"])))
    hm = folium.Map(location=[37.0902, -95.7129], #Center of USA
               zoom_start=4)
    plugins.HeatMap(lat_longs
               ).add_to(hm)
    #folium.LayerControl().add_to(hm)
    return hm



import matplotlib.pyplot as plt
def plot_LMPs_black(gpdf, LMP_, vmin, vmax):
    texas = gpd.read_file("./texas.shp")
    gpdf['LMPs'] = LMP_

    # Create a grey map of Texas
    fig, ax = plt.subplots(figsize=(10, 10))
    texas.plot(ax=ax, color='lightgrey')
    gpdf.plot(ax =ax, column = 'LMPs' , cmap = 'viridis', markersize = 5, vmin=vmin, vmax=vmax, legend=True, legend_kwds={'shrink': 0.7})
    plt.title("Map of Texas with black LMPs")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

def plot_alpha(gpdf, alpha):
    texas = gpd.read_file("./texas.shp")
    gpdf['alpha'] = alpha
    # Create a grey map of Texas
    fig, ax = plt.subplots(figsize=(10, 10))
    texas.plot(ax=ax, color='lightgrey')
    gpdf.plot(ax =ax, column = 'alpha' , cmap = 'viridis', markersize = 5, legend=True, legend_kwds={'shrink': 0.7})
    plt.title("Alpha values")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")



def plot_LMPs_green(gpdf, LMP_, lambda_green, vmin, vmax):
    texas = gpd.read_file("./texas.shp")
    gpdf['LMPs'] = np.array(LMP_) + np.array(lambda_green)
    # Create a grey map of Texas
    fig, ax = plt.subplots(figsize=(10, 10))
    texas.plot(ax=ax, color='lightgrey')
    gpdf.plot(ax =ax, column = 'LMPs' , cmap = 'viridis', markersize = 5, vmin=vmin, vmax=vmax, legend=True, legend_kwds={'shrink': 0.7})
    plt.title("Map of Texas with green LMPs")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

def get_color(line_loading):
    if line_loading >= 100:
        return 'red'
    elif line_loading > 80:
        return 'orange'
    else:
        return 'black'

def plot_lines(gpdf, from_bus, to_bus, line_loading):
    connections_df = pd.DataFrame()
    connections_df['from_bus'] = from_bus
    connections_df['to_bus'] = to_bus
    connections_df['line_loading'] = line_loading
    for index, row in connections_df.iterrows():
        from_point = gpdf[gpdf['new_bus']== row['from_bus']].geometry.values
        to_point = gpdf[gpdf['new_bus']== row['to_bus']].geometry.values
        x = [from_point.x, to_point.x]
        y = [from_point.y, to_point.y]
        color = get_color(row['line_loading'])
        plt.plot(x, y, linestyle='-', color=color) 

# def plot_lines_folium(gpdf, from_bus, to_bus, line_loading):
#     map_center = [31.9686, -99.9018]  # Center coordinates of Texas
#     map_texas = folium.Map(location=map_center, zoom_start=6, width=800, height=800)
#     connections_df = pd.DataFrame()
#     connections_df['from_bus'] = from_bus
#     connections_df['to_bus'] = to_bus
#     connections_df['line_loading'] = line_loading
#     for index, row in connections_df.iterrows():
#         from_point = [gpdf[gpdf['new_bus']== row['from_bus']].geometry.values[0].y, gpdf[gpdf['new_bus']== row['from_bus']].geometry.values[0].x]
#         to_point = [gpdf[gpdf['new_bus']== row['to_bus']].geometry.values[0].y, gpdf[gpdf['new_bus']== row['to_bus']].geometry.values[0].x]
#         color = get_color(row['line_loading'])
#         folium.PolyLine(locations=[from_point, to_point], color=color, weight = 2).add_to(map_texas)
#     legend_html = '''
#      <div style="position: fixed; 
#                  bottom: 50px; left: 50px; width: 120px; height: 100px; 
#                  border:2px solid grey; z-index:9999; font-size:14px;
#                  background-color:white;
#                  ">&nbsp; Line loadings (%) <br>
#                   &nbsp; 0 - 80 &nbsp; <i class="fa fa-square fa-1x" style="color:green"></i><br>
#                   &nbsp; 80 - 99 &nbsp; <i class="fa fa-square fa-1x" style="color:orange"></i><br>
#                   &nbsp; 100 &nbsp; <i class="fa fa-square fa-1x" style="color:red"></i>
#       </div>
#      '''
#     map_texas.get_root().html.add_child(folium.Element(legend_html))
#     return map_texas
        
def plot_lines_folium(map, gpdf, from_bus, to_bus, line_loading):
    connections_df = pd.DataFrame()
    connections_df['from_bus'] = from_bus
    connections_df['to_bus'] = to_bus
    connections_df['line_loading'] = line_loading
    polyline_layer = folium.FeatureGroup(name='Lines')
    for index, row in connections_df.iterrows():
        from_point = [gpdf[gpdf['new_bus']== row['from_bus']].geometry.values[0].y, gpdf[gpdf['new_bus']== row['from_bus']].geometry.values[0].x]
        to_point = [gpdf[gpdf['new_bus']== row['to_bus']].geometry.values[0].y, gpdf[gpdf['new_bus']== row['to_bus']].geometry.values[0].x]
        color = get_color(row['line_loading'])
        polyline = folium.PolyLine(locations=[from_point, to_point], color=color, weight = 1, popup=f"Line {int(row['from_bus'])}-{int(row['to_bus'])}<br>Line loading:{round(row['line_loading'],1)}%")
        polyline.add_to(polyline_layer)
    legend_html = '''
     <div style="position: fixed; 
                 bottom: 300px; left: 800px; width: 130px; height: 100px; 
                 border:2px solid grey; z-index:9999; font-size:14px;
                 background-color:white;
                 ">&nbsp; Line loadings (%) <br>
                  &nbsp; 0 - 80 &emsp;  <i class="fa fa-square fa-1x" style="color:rgb(0, 0, 0)"></i><br>
                  &nbsp; 80 - 99 &nbsp;  <i class="fa fa-square fa-1x" style="color:orange"></i><br>
                  &nbsp; 100 &emsp;&nbsp;&nbsp;&nbsp; <i class="fa fa-square fa-1x" style="color:red"></i>
      </div>
     '''
    map.get_root().html.add_child(folium.Element(legend_html))
    map.add_child(polyline_layer)
    # Add layer control
    # polyline_layer_control = folium.LayerControl(position='topleft')
    # polyline_layer_control.add_to(map)
    return map

def plot_green_black_gens(gpdf, generation):
    texas = gpd.read_file("./texas.shp")
    fig, ax = plt.subplots(figsize=(10, 10))
    texas.plot(ax=ax, color='lightgrey')
    merged_df = gpdf.merge(generation, left_on='new_bus', right_on='bus', how='inner')
    # merged_df['genfuel'] = genfuel
    green_gdf = merged_df[merged_df['genfuel'].isin(["wind", "hydro", "solar", "nuclear"])]
    red_gdf = merged_df[merged_df['genfuel'].isin(["coal", "ng"])]
    green_gdf.plot(ax =ax, color='green', markersize=10, label='Wind, hydro, solar, nuclear')
    red_gdf.plot(ax=ax, color='black', markersize=10, label='NG, coal')
    plt.legend()
    # Set plot title and labels
    plt.title("Map of generators in Texas")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

# def compute_line_loadings(theta, b, branch_cap, NB):
#     line_loading = np.zeros((NB, NB))
#     for n in range(NB):
#         for m in range(NB):
#             if (n != m):
#                 line_loading[n][m] = abs(b[n][m]*(theta[n]-theta[m]))/abs(branch_cap[n][m])*100
    
#     return line_loading

def compute_line_loadings(congestion_factor, theta, branch, baseMVA, NL):
    from_bus = np.array(branch['new_from_bus'])
    to_bus = np.array(branch['new_to_bus'])
    flow_limit = np.array(branch['RATE_A'] / baseMVA)
    b_lines = np.array(1 / branch['BR_X'])
    line_loading = np.zeros(NL)
    overloaded_lines = 0
    for i in range(len(b_lines)):
        from_bus_idx = int(from_bus[i])  # Convert to integer index
        to_bus_idx = int(to_bus[i])  
        line_loading[i] = abs(b_lines[i]*(theta[from_bus_idx]-theta[to_bus_idx]))/(congestion_factor*flow_limit[i])*100
        if line_loading[i] >= 100:
            overloaded_lines = overloaded_lines+1
    return line_loading, overloaded_lines

# def plot_LMPs_folium(gpdf, LMP):
#     map_center = [31.9686, -99.9018]

#     from branca.colormap import LinearColormap
#     colors = ['turquoise', 'blue', 'gray', 'black']
#     colormap = LinearColormap(colors=colors, vmin=min(LMP), vmax=max(LMP))
#     mymap = folium.Map(location=map_center, zoom_start=4)
#     gpdf['LMP'] = LMP

#     for index, row in gpdf.iterrows():
#         # Get the coordinates of the point
#         lat = row.geometry.y
#         lon = row.geometry.x
#         color = colormap(row['LMP'])
#         # Add a marker for the point
#         folium.CircleMarker(location=[lat, lon], radius=2, color=color, fill=True, popup=f"Bus: {row['new_bus']}<br>LMP:{row['LMP']}").add_to(mymap)

#     colormap.add_to(mymap)
#     return mymap

def plot_LMPs_folium(map, gpdf, LMP, vmin, vmax):

    from branca.colormap import LinearColormap
    colors = ['turquoise', 'blue', 'gray', 'black']
    # colormap = LinearColormap(colors=colors, vmin=vmin, vmax=vmax)
    gpdf['LMP'] = LMP
    marker_layer = folium.FeatureGroup(name='LMPs')

    for index, row in gpdf.iterrows():
        # Get the coordinates of the point
        lat = row.geometry.y
        lon = row.geometry.x
        # color = colormap(row['LMP'])
        # Add a marker for the point
        opacity = np.interp(row['LMP'], [abs(np.array(LMP)).min(), abs(np.array(LMP)).max()], [0, 100]) / 100
        if (row['LMP'] < 0):
            color = 'yellow'
            # color = 'red'
        elif (row['LMP'] < 10 and row['LMP'] > 0):
            color = 'darkorange'
        elif (row['LMP'] > 10 and row['LMP'] < 20):
            color = 'dodgerblue'
        else:
            color = 'darkblue'
        # circlemarker = folium.CircleMarker(location=[lat, lon], radius=7, color='blue', fill=True, fill_opaciy=opacity, opacity = opacity, popup=f"Bus: {int(row['new_bus'])}<br>LMP: {round(row['LMP'], 2)}")
        circlemarker = folium.CircleMarker(location=[lat, lon], radius=1, color=color, fill=True, popup=f"Bus: {int(row['new_bus'])}<br>LMP: {round(row['LMP'], 2)}")
        circlemarker.add_to(marker_layer)
    legend_html = '''
     <div style="position: fixed; 
                 bottom: 450px; left: 800px; width: 130px; height: 105px; 
                 border:2px solid grey; z-index:9999; font-size:14px;
                 background-color:white;
                 ">&nbsp; LMPs ($/MWh) <br>
                 &nbsp; > 20 &emsp;&emsp; <i class="fa fa-square fa-1x" style="color:rgb(0, 0, 139)"></i><br>
                 &nbsp; 10 - 20 &emsp; <i class="fa fa-square fa-1x" style="color:rgb(30, 144, 255)"></i><br>
                 &nbsp; 0 - 10 &emsp;&nbsp;&nbsp; <i class="fa fa-square fa-1x" style="color:rgb(255, 140, 0)"></i><br>
                 &nbsp; < 0 &emsp;&emsp;&nbsp;&nbsp; <i class="fa fa-square fa-1x" style="color:red"></i>
      </div>
     '''
    map.get_root().html.add_child(folium.Element(legend_html))
    map.add_child(marker_layer)
    # map.add_child(folium.LayerControl())
    # marker_layer_control = folium.LayerControl(position='topleft')
    # marker_layer_control.add_to(map)
    # colormap.add_to(map)
    return map

def plot_LMPs_folium_only_red_orange(map, gpdf, LMP, vmin, vmax):

    from branca.colormap import LinearColormap
    colors = ['turquoise', 'blue', 'gray', 'black']
    # colormap = LinearColormap(colors=colors, vmin=vmin, vmax=vmax)
    gpdf['LMP'] = LMP
    data = gpdf
    marker_layer = folium.FeatureGroup(name='LMPs')
    red_orange_data = data[(data['LMP'] < 10)]
    # red_orange_data = data[(data['LMP'] < 10) & (data['LMP'] >= 0)]
    # orange_data = data[(data['LMP'] < 10) & (data['LMP'] >= 0)]
    other_data = data[data['LMP'] >= 10]

    for index, row in other_data.iterrows():
        # Get the coordinates of the point
        lat = row.geometry.y
        lon = row.geometry.x
        # color = colormap(row['LMP'])
        # Add a marker for the point
        opacity = np.interp(row['LMP'], [abs(np.array(LMP)).min(), abs(np.array(LMP)).max()], [0, 100]) / 100
        if (row['LMP'] > 10 and row['LMP'] < 20):
            color = 'dodgerblue'
        elif (row['LMP']> 20):
            color = 'darkblue'
        # circlemarker = folium.CircleMarker(location=[lat, lon], radius=7, color='blue', fill=True, fill_opaciy=opacity, opacity = opacity, popup=f"Bus: {int(row['new_bus'])}<br>LMP: {round(row['LMP'], 2)}")
        circlemarker = folium.CircleMarker(location=[lat, lon], radius=1, color=color, fill=True, popup=f"Bus: {int(row['new_bus'])}<br>LMP: {round(row['LMP'], 2)}")
        circlemarker.add_to(marker_layer)



    for index, row in red_orange_data.iterrows():
        # Get the coordinates of the point
        lat = row.geometry.y
        lon = row.geometry.x
        # color = colormap(row['LMP'])
        # Add a marker for the point
        opacity = np.interp(row['LMP'], [abs(np.array(LMP)).min(), abs(np.array(LMP)).max()], [0, 100]) / 100
        if (row['LMP'] < 10 and row['LMP'] > 0):
            color = 'darkorange'
        elif (row['LMP'] < 0):
            color = 'yellow'
            # color = 'red'
        # circlemarker = folium.CircleMarker(location=[lat, lon], radius=7, color='blue', fill=True, fill_opaciy=opacity, opacity = opacity, popup=f"Bus: {int(row['new_bus'])}<br>LMP: {round(row['LMP'], 2)}")
        circlemarker = folium.CircleMarker(location=[lat, lon], radius=1, color=color, fill=True, popup=f"Bus: {int(row['new_bus'])}<br>LMP: {round(row['LMP'], 2)}")
        circlemarker.add_to(marker_layer)
    # html code, color red: color:red
    legend_html = '''
     <div style="position: fixed; 
                 bottom: 450px; left: 800px; width: 130px; height: 105px; 
                 border:2px solid grey; z-index:9999; font-size:14px;
                 background-color:white;
                 ">&nbsp; LMPs ($/MWh) <br>
                 &nbsp; > 20 &emsp;&emsp; <i class="fa fa-square fa-1x" style="color:rgb(0, 0, 139)"></i><br>
                 &nbsp; 10 - 20 &emsp; <i class="fa fa-square fa-1x" style="color:rgb(30, 144, 255)"></i><br>
                 &nbsp; 0 - 10 &emsp;&nbsp;&nbsp; <i class="fa fa-square fa-1x" style="color:rgb(255, 140, 0)"></i><br>
                 &nbsp; < 0 &emsp;&emsp;&nbsp;&nbsp; <i class="fa fa-square fa-1x" style="color:rgb(255,211,67)"></i>
      </div>
     '''
    map.get_root().html.add_child(folium.Element(legend_html))
    map.add_child(marker_layer)
    # map.add_child(folium.LayerControl())
    # marker_layer_control = folium.LayerControl(position='topleft')
    # marker_layer_control.add_to(map)
    # colormap.add_to(map)
    return map


def plot_green_black_LMPs_folium(map, gpdf, LMP, lambda_green, vmin , vmax):

    from branca.colormap import LinearColormap
    colors = ['turquoise', 'blue', 'gray', 'black']
    colormap = LinearColormap(colors=colors, vmin=vmin, vmax=vmax)
    LMP_green = np.array(LMP)+np.array(lambda_green)
    gpdf['LMP_green'] = LMP_green
    gpdf['LMP'] = np.array(LMP)
    marker_layer = folium.FeatureGroup(name='LMPs')
    gpdf['diff LMPs'] = abs(np.array(lambda_green))
    marker_layer2 = folium.FeatureGroup(name='LMPs - difference')
    marker_layer3 = folium.FeatureGroup(name='Green LMPs')

    for index, row in gpdf.iterrows():
        # Get the coordinates of the point
        lat = row.geometry.y
        lon = row.geometry.x
        color = colormap(row['LMP'])
        # Add a marker for the point
        opacity = np.interp(row['LMP'], [abs(np.array(LMP)).min(), abs(np.array(LMP)).max()], [50, 100]) / 100
        circlemarker = folium.CircleMarker(location=[lat, lon], radius=7, color='blue', fill=True,fill_opacity=opacity, opacity=opacity, popup=f"Bus: {int(row['new_bus'])}<br>Black LMP: {round(row['LMP'], 2)}<br>Green LMP: {round(row['LMP_green'], 2)}")
        circlemarker.add_to(marker_layer)
        opacity = np.interp(row['LMP_green'], [abs(np.array(LMP_green)).min(), abs(np.array(LMP_green)).max()], [50, 100]) / 100
        circlemarker3 = folium.CircleMarker(location=[lat, lon], radius=7, color='green', fill=True,fill_opacity=opacity, opacity=opacity, popup=f"Bus: {int(row['new_bus'])}<br>Black LMP: {round(row['LMP'], 2)}<br>Green LMP: {round(row['LMP_green'], 2)}")
        circlemarker3.add_to(marker_layer3)
        if(row['diff LMPs']>0):
            opacity = np.interp(row['diff LMPs'], [abs(np.array(lambda_green)).min(), abs(np.array(lambda_green)).max()], [50, 100]) / 100
            # apply(lambda arr: arr.apply(lambda val: (val - min_abs_value) / (max_abs_value - min_abs_value) * 100))
            circlemarker2 = folium.CircleMarker(location=[lat, lon], radius=7, color='palevioletred', fill=True, fill_opacity=opacity, opacity=opacity, popup=f"Bus: {int(row['new_bus'])}<br>Black LMP: {round(row['LMP'], 2)}<br>Green LMP: {round(row['LMP_green'], 2)}")
            circlemarker2.add_to(marker_layer2)

    map.add_child(marker_layer)
    # map.add_child(marker_layer2)
    map.add_child(marker_layer3)
    # map.add_child(folium.LayerControl())
    # marker_layer_control = folium.LayerControl(position='topleft')
    # marker_layer_control.add_to(map)
    # colormap.add_to(map)
    return map

def plot_alpha_folium(map, gpdf, alpha):
    gpdf['alpha'] = alpha
    marker_layer = folium.FeatureGroup(name='alpha')
    for index, row in gpdf.iterrows():
        # Get the coordinates of the point
        lat = row.geometry.y
        lon = row.geometry.x
        opacity = np.interp(row['alpha'], [abs(np.array(alpha)).min(), abs(np.array(alpha)).max()], [50, 100]) / 100
        circlemarker = folium.CircleMarker(location=[lat, lon], radius=7, color='palevioletred', fill=True,fill_opacity=opacity, opacity=opacity, popup=f"Bus: {int(row['new_bus'])}<br>Alpha: {round(row['alpha'], 2)}")
        circlemarker.add_to(marker_layer)

    map.add_child(marker_layer)
    
    return map

def plot_green_black_gen_folium(map, gpdf, generation):
    merged_df = gpdf.merge(generation, left_on='new_bus', right_on='bus', how='inner')
    marker_layer_gens = folium.FeatureGroup(name='Generation')
    for index, row in merged_df.iterrows():
        # Determine color based on genfuel
        if row['genfuel'] in ["wind", "hydro", "solar", "nuclear"]:
            color = 'lawngreen'
        elif row['genfuel'] in ["coal", "ng"]:
            color = 'black'
        else:
            color = 'gray'  # Default color for other fuels
            
        lat = row.geometry.y
        lon = row.geometry.x
         # Add a marker for the point
        circlemarker = folium.CircleMarker(location=[lat, lon], radius=7, color=color, fill=True, popup=f"Bus: {int(row['new_bus'])}<br>Generation: {row['genfuel']}")
        circlemarker.add_to(marker_layer_gens)
    map.add_child(marker_layer_gens)
    # marker_layer_control = folium.LayerControl(position='topleft')
    # marker_layer_control.add_to(map)
    return map

def plot_load_bids_folium(map, gpdf, loads, load_bids, alpha, NB, baseMVA):
    from branca.colormap import LinearColormap
    colors = ['tan', 'khaki', 'olive', 'darkolivegreen']
    # colors2 = ['lightblue', 'cyan', 'darkturquoise', 'darkcyan']
    # colors = ['tan', 'mediumslateblue', 'rebeccapurple', 'indigo']
    colormap = LinearColormap(colors=colors, vmin=min(loads['PD']/baseMVA), vmax=max(loads['PD']/baseMVA))
    # colormap2 = LinearColormap(colors=colors2, vmin=min(alpha), vmax=max(alpha))
    # colormap3 = LinearColormap(colors=colors3, vmin=min(load_bids), vmax=max(load_bids))
    load_df = pd.DataFrame()
    load_df['bus'] = range(NB)
    load_df['PD'] = loads['PD'].reset_index(drop=True)
    load_df['alpha'] = alpha
    load_df['bids'] = load_bids
    merged_df = gpdf.merge(load_df, left_on='new_bus', right_on='bus', how='left')
    marker_layer_loads = folium.FeatureGroup(name='Loads')
    for index, row in merged_df.iterrows():
        # Determine color based on genfuel
        if row['PD'] > 0.01:
            lat = row.geometry.y
            lon = row.geometry.x
            color = colormap(row['PD']/baseMVA)
            # Add a marker for the point
            circlemarker = folium.CircleMarker(location=[lat, lon], radius=3, color=color, fill=True, popup=f"Bus: {int(row['new_bus'])}<br>Demand: {row['PD']/baseMVA}<br>Bid: {round(row['bids'], 2)}<br>Alpha: {round(row['alpha'], 2)}")
            circlemarker.add_to(marker_layer_loads)
    map.add_child(marker_layer_loads)
    # marker_layer_control = folium.LayerControl(position='topleft')
    # marker_layer_control.add_to(map)
    colormap.add_to(map)
    return map, merged_df

import math
def plot_green_not_dispatched(map, gpdf, green_generation,pg_green_):
    marker_layer_green_gens = folium.FeatureGroup(name='Green generation not dispatched')
    dispatch = pd.DataFrame()
    dispatch['green_not_dispatched'] = green_generation['max_p_mw'] - pg_green_
    dispatch['bus'] = green_generation['bus'] 
    green_not_dispatched = dispatch[dispatch['green_not_dispatched'] > 0]
    merged_df = gpdf.merge(green_not_dispatched, left_on='new_bus', right_on='bus', how='left')
    for index, row in merged_df.iterrows():
        if not math.isnan(row['green_not_dispatched']):
            lat = row.geometry.y
            lon = row.geometry.x
            circlemarker = folium.CircleMarker(location=[lat, lon], radius=4, color='brown', fill=False, popup=f"Bus: {int(row['new_bus'])}<br>Generation not dispatched: {round(row['green_not_dispatched'], 2)}")
            circlemarker.add_to(marker_layer_green_gens)
    map.add_child(marker_layer_green_gens)    
    marker_layer_control = folium.LayerControl(position='topleft')
    marker_layer_control.add_to(map)
    return map