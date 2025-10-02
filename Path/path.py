import pandas as pd
import math
import heapq
from typing import List, Dict, Tuple, Optional
import numpy as np

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth using the Haversine formula.
    
    Parameters:
    lat1, lon1: Latitude and longitude of first point (in decimal degrees)
    lat2, lon2: Latitude and longitude of second point (in decimal degrees)
    
    Returns:
    Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth's radius in kilometers
    earth_radius = 6371.0
    
    return earth_radius * c

class AStarPathfinder:
    """
    A* pathfinding implementation for POI route optimization using Haversine distance.
    """
    
    def __init__(self, poi_df: pd.DataFrame):
        """
        Initialize the pathfinder with POI data.
        
        Parameters:
        poi_df: DataFrame with columns ['id', 'latitude', 'longitude'] at minimum
        """
        self.poi_df = poi_df.copy()
        self.nodes = {}  # id -> (lat, lon)
        self._build_node_dict()
    
    def _build_node_dict(self):
        """Build internal dictionary for fast node lookup."""
        for _, row in self.poi_df.iterrows():
            self.nodes[row['id']] = (row['latitude'], row['longitude'])
    
    def _heuristic(self, node_id: int, goal_id: int) -> float:
        """
        Heuristic function for A*. Uses Haversine distance as an admissible heuristic.
        
        Parameters:
        node_id: Current node ID
        goal_id: Goal node ID
        
        Returns:
        Estimated cost (Haversine distance in km)
        """
        lat1, lon1 = self.nodes[node_id]
        lat2, lon2 = self.nodes[goal_id]
        return haversine_distance(lat1, lon1, lat2, lon2)
    
    def _get_neighbors(self, node_id: int, valid_nodes: set) -> List[int]:
        """
        Get all valid neighboring nodes (all other nodes in the valid set).
        
        Parameters:
        node_id: Current node ID
        valid_nodes: Set of valid node IDs to consider
        
        Returns:
        List of neighboring node IDs
        """
        return [neighbor for neighbor in valid_nodes if neighbor != node_id]
    
    def _reconstruct_path(self, came_from: Dict[int, int], current: int) -> List[int]:
        """
        Reconstruct the optimal path from the came_from dictionary.
        
        Parameters:
        came_from: Dictionary mapping node_id -> parent_node_id
        current: Current (goal) node ID
        
        Returns:
        List of node IDs representing the path from start to goal
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Reverse to get start->goal order
    
    def find_shortest_path(self, start_id: int, goal_id: int, 
                          valid_nodes: Optional[set] = None) -> Tuple[List[int], float]:
        """
        Find the shortest path between two POIs using A* algorithm.
        
        Parameters:
        start_id: Starting POI ID
        goal_id: Goal POI ID
        valid_nodes: Optional set of valid node IDs to consider (if None, uses all nodes)
        
        Returns:
        Tuple of (path_as_list_of_ids, total_distance_km)
        """
        if valid_nodes is None:
            valid_nodes = set(self.nodes.keys())
        
        if start_id not in valid_nodes or goal_id not in valid_nodes:
            raise ValueError("Start or goal node not in valid nodes set")
        
        if start_id == goal_id:
            return [start_id], 0.0
        
        # A* algorithm implementation
        open_set = [(0, start_id)]  # Priority queue: (f_score, node_id)
        came_from = {}  # node_id -> parent_node_id
        g_score = {start_id: 0}  # Actual cost from start
        f_score = {start_id: self._heuristic(start_id, goal_id)}  # g + h
        
        open_set_hash = {start_id}  # For efficient membership testing
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            if current == goal_id:
                path = self._reconstruct_path(came_from, current)
                return path, g_score[current]
            
            for neighbor in self._get_neighbors(current, valid_nodes):
                # Calculate actual distance (cost) to neighbor
                lat1, lon1 = self.nodes[current]
                lat2, lon2 = self.nodes[neighbor]
                tentative_g = g_score[current] + haversine_distance(lat1, lon1, lat2, lon2)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_id)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        # No path found
        raise ValueError(f"No path found from {start_id} to {goal_id}")

def find_optimal_path(poi_list: pd.DataFrame, start_id: int, end_id: int, 
                     intermediate_pois: Optional[List[int]] = None) -> Tuple[List[Dict], float]:
    """
    Main function to find optimal path between POIs using A* algorithm.
    
    Parameters:
    poi_list: DataFrame containing POI data with columns ['id', 'latitude', 'longitude']
    start_id: Starting POI ID
    end_id: Ending POI ID
    intermediate_pois: Optional list of intermediate POI IDs to visit
    
    Returns:
    Tuple of (path_data_as_list_of_dicts, total_distance_km)
    where path_data contains full POI information for each step
    """
    # Validate required columns
    required_columns = ['id', 'latitude', 'longitude']
    if not all(col in poi_list.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    pathfinder = AStarPathfinder(poi_list)
    
    if intermediate_pois is None or len(intermediate_pois) == 0:
        # Simple case: direct path from start to end
        path_ids, total_distance = pathfinder.find_shortest_path(start_id, end_id)
    else:
        # Complex case: visit intermediate POIs
        # This is a simplified approach - for optimal results with many intermediate points,
        # consider using TSP solver or more sophisticated route optimization
        
        all_waypoints = [start_id] + intermediate_pois + [end_id]
        path_ids = []
        total_distance = 0.0
        
        for i in range(len(all_waypoints) - 1):
            current_start = all_waypoints[i]
            current_end = all_waypoints[i + 1]
            
            segment_path, segment_distance = pathfinder.find_shortest_path(current_start, current_end)
            
            # Avoid duplicating waypoints between segments
            if i > 0:
                segment_path = segment_path[1:]  # Remove first element (duplicate)
            
            path_ids.extend(segment_path)
            total_distance += segment_distance
    
    # Convert path IDs to full POI data
    path_data = []
    for poi_id in path_ids:
        poi_row = poi_list[poi_list['id'] == poi_id].iloc[0].to_dict()
        path_data.append(poi_row)
    
    return path_data, total_distance

def find_optimal_marketing_route(poi_list: pd.DataFrame, target_poi_ids: List[int], 
                                start_id: Optional[int] = None, end_id: Optional[int] = None) -> Dict:
    """
    Advanced function to find optimal marketing route visiting multiple POIs.
    
    Parameters:
    poi_list: DataFrame containing POI data
    target_poi_ids: List of POI IDs that must be visited
    start_id: Optional starting point (if None, uses first POI in target list)
    end_id: Optional ending point (if None, uses last POI in target list)
    
    Returns:
    Dictionary containing route information including path, distances, and summary stats
    """
    if len(target_poi_ids) < 2:
        raise ValueError("Need at least 2 POIs for route optimization")
    
    if start_id is None:
        start_id = target_poi_ids[0]
        intermediate_pois = target_poi_ids[1:-1] if len(target_poi_ids) > 2 else []
        end_id = target_poi_ids[-1] if end_id is None else end_id
    else:
        intermediate_pois = [poi for poi in target_poi_ids if poi not in [start_id, end_id]]
        if end_id is None:
            end_id = target_poi_ids[-1]
    
    path_data, total_distance = find_optimal_path(poi_list, start_id, end_id, intermediate_pois)
    
    # Calculate additional statistics
    num_pois_visited = len(path_data)
    avg_distance_per_segment = total_distance / (num_pois_visited - 1) if num_pois_visited > 1 else 0
    
    # Estimate travel time (assuming average speed of 40 km/h in urban areas)
    estimated_travel_time_hours = total_distance / 40.0
    
    return {
        'route_path': path_data,
        'total_distance_km': round(total_distance, 2),
        'num_pois_visited': num_pois_visited,
        'avg_distance_per_segment_km': round(avg_distance_per_segment, 2),
        'estimated_travel_time_hours': round(estimated_travel_time_hours, 2),
        'start_poi': path_data[0] if path_data else None,
        'end_poi': path_data[-1] if path_data else None
    }

# Example usage and testing function
def demo_pathfinding():
    """
    Demonstration of the pathfinding algorithm with sample data.
    """
    # Create sample POI data (using your provided example)
    sample_data = {
        'id': [29566, 29567, 29568, 29569, 29570],
        'kabupaten_kota': ['Kota Bandung'] * 5,
        'name': ['Setrasari Mall', 'Setiabudhi Supermarket', 'Circle K', 'Premier Cihampelas', 'Circle K'],
        'category': ['mall', 'supermarket', 'convenience', 'supermarket', 'convenience'],
        'latitude': [-6.880704, -6.882407, -6.885564, -6.896159, -6.902086],
        'longitude': [107.582993, 107.601281, 107.611822, 107.603597, 107.611388]
    }
    
    poi_df = pd.read_csv('../osm_bandung_30rows.csv')
    
    print("=== A* Pathfinding Demo ===")
    print(f"Sample POI data:\n{poi_df[['id', 'name', 'category', 'latitude', 'longitude']]}")
    print()
    
    # Test 1: Simple path between two points
    print("Test 1: Direct path from Setrasari Mall to Circle K (29570)")
    try:
        path_data, distance = find_optimal_path(poi_df, 29566, 29570)
        print(f"Path found: {[poi['name'] for poi in path_data]}")
        print(f"Total distance: {distance:.2f} km")
        print()
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Marketing route visiting multiple POIs
    print("Test 2: Marketing route visiting multiple POIs")
    try:
        target_pois = [29566, 29567, 29568, 29569]
        route_info = find_optimal_marketing_route(poi_df, target_pois)
        
        print("Route Summary:")
        print(f"- Total distance: {route_info['total_distance_km']} km")
        print(f"- POIs visited: {route_info['num_pois_visited']}")
        print(f"- Estimated travel time: {route_info['estimated_travel_time_hours']:.1f} hours")
        print(f"- Route: {' -> '.join([poi['name'] for poi in route_info['route_path']])}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    demo_pathfinding()