import subprocess


def main():
    res = []
    for num_threads in range(1, 17):
        p = subprocess.Popen(["make", 
            "NB_THREADS={}".format(num_threads)], stdout=subprocess.PIPE)
        p.wait()
        out = subprocess.check_output(
            "./mandelbrot-256-500-375--2-0.6--1-1-{}-0".format(num_threads)
        ).decode("UTF-8")

        time = float(out.split(" ")[1].strip())
        print(time)


if __name__ == '__main__':
    main()

