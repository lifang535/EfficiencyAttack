def monitor_queues(queues, names, interval=1):
    """
    Monitor process that updates the terminal in-place with the sizes of the given queues.
    
    This uses ANSI escape codes to move the cursor back up to the previously printed lines.
    """
    previous_lines = 0
    while True:
        output_lines = []
        output_lines.append("📦  Current Queue Sizes:")
        for q, name in zip(queues, names):
            try:
                size = q.qsize()
            except NotImplementedError:
                size = 'N/A'
            output_lines.append(f"  {name:15}: {size}")
        output_lines.append("-" * 30)
        
        # Move cursor up to overwrite the previous output (if any)
        if previous_lines:
            sys.stdout.write("\033[F" * previous_lines)
        # Join and print new output
        output = "\n".join(output_lines)
        print(output)
        sys.stdout.flush()
        previous_lines = len(output_lines)
        time.sleep(interval)
