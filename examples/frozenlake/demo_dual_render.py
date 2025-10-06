"""
Demo script to show how to use the dual rendering mode in FrozenLake.

The dual rendering mode returns both text and image observations in a single call.
This is useful when you want to log or visualize both representations simultaneously.
"""


def demo_dual_render():
    """
    Example of using the dual rendering mode.
    
    The render(mode="dual") method returns a dictionary with:
    - "text": The text-based observation (same as "tiny_rgb_array")
    - "image": The RGB image array from gymnasium (same as "rgb_array")
    - "state": The state representation as a numpy array
    - "list": A list representation of the grid
    """
    
    print("=" * 70)
    print("FrozenLake Dual Rendering Demo")
    print("=" * 70)
    
    # Example usage in code:
    code_example = """
    from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv
    
    # Create environment
    env = FrozenLakeEnv(size=4, seed=42, is_slippery=False)
    
    # Reset environment
    obs, info = env.reset()
    
    # Get both text and image observations
    dual_obs = env.render(mode="dual")
    
    # Access different representations
    text_obs = dual_obs["text"]      # Text grid representation
    image_obs = dual_obs["image"]    # RGB image array (H, W, 3)
    state_obs = dual_obs["state"]    # State numpy array
    list_obs = dual_obs["list"]      # List representation
    
    # You can also get individual modes
    text_only = env.render(mode="tiny_rgb_array")  # Default
    image_only = env.render(mode="rgb_array")      # Image only
    state_only = env.render(mode="state")          # State only
    """
    
    print("\nExample Code:")
    print(code_example)
    
    print("\n" + "=" * 70)
    print("Available Render Modes:")
    print("=" * 70)
    print("1. 'tiny_rgb_array' - Text-based grid representation (default)")
    print("2. 'rgb_array'      - RGB image array from gymnasium")
    print("3. 'ansi'           - ANSI colored text representation")
    print("4. 'state'          - Numpy array of state integers")
    print("5. 'list'           - List representation of the grid")
    print("6. 'dual'           - Dictionary with all representations")
    
    print("\n" + "=" * 70)
    print("Dual Mode Output Structure:")
    print("=" * 70)
    print("""
    {
        "text": <string>,           # Text grid like " P \\t _ \\t O \\t G \\t\\n..."
        "image": <np.array>,        # RGB image array with shape (H, W, 3)
        "state": <np.array>,        # Integer state array with shape (size, size)
        "list": <list[str]>,        # List of strings like ["P _ O G", ...]
    }
    """)
    
    print("\n" + "=" * 70)
    print("Use Cases:")
    print("=" * 70)
    print("✓ Logging both text and visual representations simultaneously")
    print("✓ Training agents on images while debugging with text")
    print("✓ Creating visualizations that show both formats side-by-side")
    print("✓ Saving observations in multiple formats for analysis")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_dual_render()


