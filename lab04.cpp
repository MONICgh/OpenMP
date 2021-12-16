#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <omp.h>

using namespace std;

#define CHUNK_SIZE 8
#define KIND static

int flows;

struct RGB {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};


void read_P5 (FILE* fin, FILE* fout, int &h, int &w, int &max_color,
    vector <vector <unsigned char>> &picsels) {

    fscanf(fin, "%d %d", &h, &w);
    fscanf(fin, "%d\n", &max_color);

    fprintf(fout, "%d %d ", h, w);
    fprintf(fout, "%d\n", max_color);

    picsels.resize(h, vector <unsigned char> (w));

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            fscanf(fin, "%c", &picsels[i][j]);
        }
    }
}

void read_P6 (FILE* fin, FILE* fout, int &h, int &w, int &max_color,
    vector <vector <RGB>> &picsels) {

    fscanf(fin, "%d %d", &h, &w);
    fscanf(fin, "%d\n", &max_color);

    fprintf(fout, "%d %d ", h, w);
    fprintf(fout, "%d\n", max_color);

    picsels.resize(h, vector <RGB> (w));

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            fscanf(fin, "%c", &picsels[i][j].r);
            fscanf(fin, "%c", &picsels[i][j].g);
            fscanf(fin, "%c", &picsels[i][j].b);
        }
    }
}

void treatment_P5 (FILE* fin, FILE* fout) {

    int h, w, max_color;
    vector <vector <unsigned char>> pixels;

    read_P5 (fin, fout, h, w, max_color, pixels);

    vector <unsigned char> mns(flows, 255);
    vector <unsigned char> mxs(flows);

    unsigned char mn = 255;
    unsigned char mx = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    omp_set_num_threads(flows);
    #pragma omp parallel
    {
        #pragma omp for schedule(KIND, CHUNK_SIZE)
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                mns[omp_get_thread_num()] = min(mns[omp_get_thread_num()], pixels[i][j]);
                mxs[omp_get_thread_num()] = max(mxs[omp_get_thread_num()], pixels[i][j]);
            }
        }
    }

    for (int flow = 0; flow < flows; flow++) {
        mn = min(mn, mns[flow]);
        mx = max(mx, mxs[flow]);
    }

    auto f = [&] (unsigned char x) {
        return (unsigned char) (int(x - mn) * 255 / int(mx - mn));
    };

    omp_set_num_threads(flows);
    #pragma omp parallel
    {
        #pragma omp for schedule(KIND, CHUNK_SIZE)
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                pixels[i][j] = f(pixels[i][j]);
            }
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    cerr << "Time (" << flows << " thread(s)): ";
    cerr << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " ms\n";


    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            fprintf(fout, "%c", pixels[i][j]);
        }
    }

}

void treatment_P6 (FILE* fin, FILE* fout) {
    int h, w, max_color;
    vector <vector <RGB>> pixels;
    
    read_P6 (fin, fout, h, w, max_color, pixels);

    vector <unsigned char> mns(flows, 255);
    vector <unsigned char> mxs(flows);

    unsigned char mn = 255;
    unsigned char mx = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    omp_set_num_threads(flows);
    #pragma omp parallel
    {
        #pragma omp for schedule(KIND, CHUNK_SIZE)
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                mns[omp_get_thread_num()] = min(mns[omp_get_thread_num()], pixels[i][j].r);
                mns[omp_get_thread_num()] = min(mns[omp_get_thread_num()], pixels[i][j].g);
                mns[omp_get_thread_num()] = min(mns[omp_get_thread_num()], pixels[i][j].b);
                mxs[omp_get_thread_num()] = max(mxs[omp_get_thread_num()], pixels[i][j].r);
                mxs[omp_get_thread_num()] = max(mxs[omp_get_thread_num()], pixels[i][j].g);
                mxs[omp_get_thread_num()] = max(mxs[omp_get_thread_num()], pixels[i][j].b);
            }
        }
    }

    for (int flow = 0; flow < flows; flow++) {
        mn = min(mn, mns[flow]);
        mx = max(mx, mxs[flow]);
    }

    auto f = [&] (unsigned char x) {
        return (unsigned char) (int(x - mn) * 255 / int(mx - mn));
    };

    omp_set_num_threads(flows);
    #pragma omp parallel
    {
        #pragma omp for schedule(KIND, CHUNK_SIZE)
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                pixels[i][j].r = f(pixels[i][j].r);
                pixels[i][j].g = f(pixels[i][j].g);
                pixels[i][j].b = f(pixels[i][j].b);
            }
        }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    cerr << "Time (" << flows << " thread(s)): ";
    cerr << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " ms\n";

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            fprintf(fout, "%c", pixels[i][j].r);
            fprintf(fout, "%c", pixels[i][j].g);
            fprintf(fout, "%c", pixels[i][j].b);
        }
    }

}

void plus_P5 (FILE* fin, FILE* fout, double k) {

    int h, w, max_color;
    vector <vector <unsigned char>> picsels;

    read_P5 (fin, fout, h, w, max_color, picsels);

    vector <vector <int>> srt (flows, vector <int> (255));

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    omp_set_num_threads(flows);
    #pragma omp parallel
    {
        #pragma omp for schedule(KIND, CHUNK_SIZE)
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) srt[omp_get_thread_num()][picsels[i][j]]++;
        }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    cerr << "Time (" << flows << " thread(s)): ";
    cerr << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " ms\n";

    int sz = h * w;
    unsigned char mn = 255;
    unsigned char mx = 0;
    
    int pxls = 0;
    for (int i = 0; i <= 255; i++) {
        for (int flow = 0; flow < flows; flow++) pxls += srt[flow][i];
        if ((double) pxls > (double) sz * k) {
            mn = i;
            break;
        }
    }

    pxls = 0;
    for (int i = 255; i >= 0; i--) {
        for (int flow = 0; flow < flows; flow++) pxls += srt[flow][i];
        if ((double) pxls > (double) sz * k) {
            mx = i;
            break;
        }
    }

    auto f = [&] (unsigned char x) {
        int pxl = (int(x - mn) * 255 / int(mx - mn));
        if (pxl < 0) return (unsigned char) 0;
        if (pxl > 255) return (unsigned char) 255;
        return (unsigned char)(int(x - mn) * 255 / int(mx - mn));
    };

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            fprintf(fout, "%c", f(picsels[i][j]));
        }
    }
}

void plus_P6 (FILE* fin, FILE* fout, double k) {

    int h, w, max_color;
    vector <vector <RGB>> pixels;

    read_P6 (fin, fout, h, w, max_color, pixels);

    vector <vector <int>> srt (flows, vector <int> (255));

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    omp_set_num_threads(flows);
    #pragma omp parallel
    {
        #pragma omp for schedule(KIND, CHUNK_SIZE)
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                srt[omp_get_thread_num()][pixels[i][j].r]++;
                srt[omp_get_thread_num()][pixels[i][j].g]++;
                srt[omp_get_thread_num()][pixels[i][j].b]++;
            }
        }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    cerr << "Time (" << flows << " thread(s)): ";
    cerr << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " ms\n";

    int sz = 3 * h * w;
    unsigned char mn = 255;
    unsigned char mx = 0;
    
    int pxls = 0;
    for (int i = 0; i <= 255; i++) {
        for (int flow = 0; flow < flows; flow++) {
            pxls += srt[flow][i];
        }
        if ((double) pxls > (double) sz * k) {
            mn = i;
            break;
        }
    }

    pxls = 0;
    for (int i = 255; i >= 0; i--) {
        for (int flow = 0; flow < flows; flow++) pxls += srt[flow][i];
        if ((double) pxls > (double) sz * k) {
            mx = i;
            break;
        }
    }

    auto f = [&] (unsigned char x) {
        int pxl = (int(x - mn) * 255 / int(mx - mn));
        if (pxl < 0) return (unsigned char) 0;
        if (pxl > 255) return (unsigned char) 255;
        return (unsigned char)(int(x - mn) * 255 / int(mx - mn));
    };

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            fprintf(fout, "%c", f(pixels[i][j].r));
            fprintf(fout, "%c", f(pixels[i][j].g));
            fprintf(fout, "%c", f(pixels[i][j].b));
        }
    }
}


int main (int argc, char const* argv[]) {

    flows = atoi(argv[1]);

    char const* fin_name = argv[2];
    char const* fout_name = argv[3];

    FILE* fin = fopen(fin_name, "rb");
    FILE* fout = fopen(fout_name, "wb");

    char type[2];
    fscanf(fin, "%c%c", &type[0], &type[1]);
    fprintf(fout, "%c%c ", type[0], type[1]);

    if (argc == 4) {
        if (type[1] == '5') treatment_P5(fin, fout);
        else treatment_P6(fin, fout);
    }
        
    else {
        double k = stod (argv[4]);
        if (type[1] == '5') plus_P5(fin, fout, k);
        else plus_P6(fin, fout, k);
    }

    fclose(fin);
    fclose(fout);

    return 0;
}