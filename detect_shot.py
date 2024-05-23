def detect_shot(rim_y : int, ball_y : int, height_threshold : int = 100):
    if ball_y < rim_y - height_threshold:
        return True
    return False
    

