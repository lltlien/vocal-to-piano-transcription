def add_rests(notes, intervals):
    notes_rest = []
    for i in range(1, len(notes)):
        start_time = intervals[i-1][1]
        end_time = intervals[i][0]
        if end_time - start_time >= 0:
            notes_rest.append((0, start_time, end_time))
    
    notes += [0] * len(notes_rest)
    intervals += [(start_time, end_time) for _, start_time, end_time in notes_rest]
    
    combined = sorted(zip(notes, intervals), key=lambda x: x[1][0])
    sorted_notes, sorted_intervals = zip(*combined)
    print(sorted_notes, sorted_intervals)
    
    return list(sorted_notes), list(sorted_intervals)