import time
from toxiproxy import Toxiproxy

# Connect to the Toxiproxy server
toxiproxy = Toxiproxy()

# Define the proxy
proxy = toxiproxy.create_proxy(
    name="video_stream",
    listen="tcp://0.0.0.0:8666",
    upstream="tcp://127.0.0.1:9999"
)

# Define the network condition presets
NETWORK_PRESETS = [
    {"name": "VERY POOR", "latency": 300, "jitter": 50, "bandwidth_rate": 500},
    {"name": "POOR", "latency": 150, "jitter": 30, "bandwidth_rate": 1000},
    {"name": "FAIR", "latency": 80, "jitter": 20, "bandwidth_rate": 2000},
    {"name": "GOOD", "latency": 40, "jitter": 10, "bandwidth_rate": 5000},
    {"name": "EXCELLENT", "latency": 20, "jitter": 5, "bandwidth_rate": 10000},
    {"name": "ULTRA", "latency": 1, "jitter": 0, "bandwidth_rate": 50000}
]

def apply_conditions(preset):
    """Applies a network condition preset to the proxy."""
    print(f"Applying preset: {preset['name']}")
    proxy.update_toxic("latency", latency=preset["latency"], jitter=preset["jitter"])
    proxy.update_toxic("bandwidth", rate=preset["bandwidth_rate"])

def main():
    """Cycles through the network condition presets."""
    # Add initial toxics
    proxy.add_toxic("latency", "latency", latency=0)
    proxy.add_toxic("bandwidth", "bandwidth", rate=100000) # Start with high bandwidth

    try:
        while True:
            for preset in NETWORK_PRESETS:
                apply_conditions(preset)
                time.sleep(20)
    except KeyboardInterrupt:
        print("\nResetting proxy and exiting.")
        proxy.delete()

if __name__ == "__main__":
    main()