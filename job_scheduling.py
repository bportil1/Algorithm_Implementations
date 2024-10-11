def schedule_jobs(jobs):
    # Each job is a tuple (pi, fi)
    # Compute ratios and sort jobs
    #jobs.sort(key=lambda x: x[1] / x[0], reverse=True)
    jobs.sort(key=lambda x: x[1], reverse=True)

    print(jobs)

    current_time = 0
    finish_times = []

    for pi, fi in jobs:
        current_time += pi  # Time spent on the supercomputer
        finish_time = current_time + fi  # Time when job will finish on PC
        finish_times.append(finish_time)
    print(finish_times)
    # The completion time is the maximum of the finish times
    return max(finish_times)

jobs=[(2,5), (3,5), (7,9)]

jobs1=[(5,2), (5,3), (9,7)]

jobs3=[(5,2), (5,3), (9,1)]

jobs4=[(10,2), (5,3), (1,9)]


print(schedule_jobs(jobs))
print(schedule_jobs(jobs1))
print(schedule_jobs(jobs3))
print(schedule_jobs(jobs4))
