#shot logic:
# - keep track of latest player within certain distance of ball (might need another model to track ball)
# - if ball is a certain threshold above rim, it is a shot
def detect_shot(rim_y : int, ball_y : int, height_threshold : int = 100):
    if ball_y < rim_y - height_threshold:
        return True
    return False
    

