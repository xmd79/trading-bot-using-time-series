            ##################################################
            ##################################################

            with open("signals.txt", "a") as f:   
                # Get data and calculate indicators here...
                timestamp = current_time.strftime("%d %H %M %S")

                if price < fast_target1 < fast_target2 < fast_target3 < fast_target4 and price < fastest_target and price < target1 < target2 < target3 < target4 < target5:
                        if price < avg_mtf and dist_from_close_to_min < 15 and current_quadrant == 1:
                            if tr_sig == "Buy" or tr_sig == "Hold":
                                if momentum > 0:
                                    trigger_long = True

                if price > fast_target1 > fast_target2 > fast_target3 > fast_target4 and price > fastest_target and price > target1 > target2 > target3 > target4 > target5:
                        if price > avg_mtf and dist_from_close_to_max < 5 and current_quadrant == 4:
                            if tr_sig == "Sell" or tr_sig == "Hold":
                                if momentum < 0:
                                    trigger_short = True  
                if trigger_long:          
                    print("LONG signal!")  
                    f.write(f"{timestamp} LONG {price}\n") 
                    trigger_long = False
         
                if trigger_short:
                    print("SHORT signal!")
                    f.write(f"{timestamp} SHORT {price}\n")
                    trigger_short = False

                ##################################################
                ##################################################
