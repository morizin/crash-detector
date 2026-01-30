def hook(frame_data, context):
    print(f"{frame_data = } \n\n\n, {context = }")
    frame = frame_data[
        "modified"
    ]  # Using 'modified' to propagate changes from possible previous stages in the frame execution path
    # Do something to the frame here
    # ...

    # If you did not modify the frame in place update it
    frame_data["modified"] = frame
