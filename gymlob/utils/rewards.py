def get_step_reward(step_executed_orders, order_direction, arrival_price):

    step_reward = 0
    for q_executed, p_executed in step_executed_orders:
        r = 0
        if order_direction == 'BUY':
            r = q_executed * (arrival_price - p_executed)
        elif order_direction == 'SELL':
            r = q_executed * (p_executed - arrival_price)
        step_reward += r

    return step_reward