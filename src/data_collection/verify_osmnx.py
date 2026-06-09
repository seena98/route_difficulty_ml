import os
import osmnx as ox

def test_osmnx():
    print("Testing OSMnx installation...")
    try:
        # Download a tiny drivable network around the Brandenburg Gate
        location = "Brandenburg Gate, Berlin, Germany"
        print(f"Downloading drivable road network within 500m of {location}...")
        
        # Get graph
        G = ox.graph_from_address(location, dist=500, network_type="drive")
        
        # Check nodes and edges
        num_nodes = len(G.nodes)
        num_edges = len(G.edges)
        print(f"Success! Downloaded graph has {num_nodes} nodes and {num_edges} edges.")
        
        # Try writing to a GraphML file
        output_dir = "data/temp"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, "test_network.graphml")
        ox.save_graphml(G, filepath)
        print(f"Successfully saved test graph to {filepath}")
        print("OSMnx installation is fully functional!")
        return True
    except Exception as e:
        print(f"Error during OSMnx verification: {e}")
        return False

if __name__ == "__main__":
    test_osmnx()
