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
const int SZ = 256;

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

    unsigned char mn = 255;
    unsigned char mx = 0;

    auto f = [&] (unsigned char x) {
        return (unsigned char) (int(x - mn) * SZ / int(mx - mn + 1));
    };

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    if (flows == 0) {

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                mn = min(mn, pixels[i][j]);
                mx = max(mx, pixels[i][j]);
            }
        }

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                pixels[i][j] = f(pixels[i][j]);
            }
        }

    }
    else {

        vector <unsigned char> mns(flows, 255);
        vector <unsigned char> mxs(flows);

        omp_set_num_threads(flows);
        #pragma omp parallel for schedule(KIND, CHUNK_SIZE) shared(mns, mxs)
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                mns[omp_get_thread_num()] = min(mns[omp_get_thread_num()], pixels[i][j]);
                mxs[omp_get_thread_num()] = max(mxs[omp_get_thread_num()], pixels[i][j]);
            }
        }

        for (int flow = 0; flow < flows; flow++) {
            mn = min(mn, mns[flow]);
            mx = max(mx, mxs[flow]);
        }

        omp_set_num_threads(flows);
        #pragma omp parallel for schedule(KIND, CHUNK_SIZE) shared(pixels)
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

    unsigned char mn = 255;
    unsigned char mx = 0;

    auto f = [&] (unsigned char x) {
        return (unsigned char) (int(x - mn) * SZ / int(mx - mn + 1));
    };

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    if (flows == 0) {

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                mn = min(mn, pixels[i][j].r);
                mn = min(mn, pixels[i][j].g);
                mn = min(mn, pixels[i][j].b);
                mx = max(mx, pixels[i][j].r);
                mx = max(mx, pixels[i][j].g);
                mx = max(mx, pixels[i][j].b);
            }
        }

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                pixels[i][j].r = f(pixels[i][j].r);
                pixels[i][j].g = f(pixels[i][j].g);
                pixels[i][j].b = f(pixels[i][j].b);
            }
        }

    }
    else {

        vector <unsigned char> mns(flows, 255);
        vector <unsigned char> mxs(flows);

        omp_set_num_threads(flows);
        #pragma omp parallel for schedule(KIND, CHUNK_SIZE) shared(mns, mxs)
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

        for (int flow = 0; flow < flows; flow++) {
            mn = min(mn, mns[flow]);
            mx = max(mx, mxs[flow]);
        }

        omp_set_num_threads(flows);
        #pragma omp parallel for schedule(KIND, CHUNK_SIZE) shared(pixels)
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
    vector <vector <unsigned char>> pixels;

    read_P5 (fin, fout, h, w, max_color, pixels);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    int sz = h * w;
    unsigned char mn = 255;
    unsigned char mx = 0;

    auto f = [&] (unsigned char x) {
        int pxl = (int(x - mn) * SZ / int(mx - mn + 1));
        if (pxl < 0) return (unsigned char) 0;
        if (pxl > 255) return (unsigned char) 255;
        return (unsigned char) pxl;
    };


    if (flows == 0) {

        vector <int> srt (SZ);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) srt[pixels[i][j]]++;
        }

        int pxls = 0;
        for (int i = 0; i <= 255; i++) {
            pxls += srt[i];
            if ((double) pxls > (double) sz * k) {
                mn = i;
                break;
            }
        }

        pxls = 0;
        for (int i = 255; i >= 0; i--) {
            pxls += srt[i];
            if ((double) pxls > (double) sz * k) {
                mx = i;
                break;
            }
        }

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                pixels[i][j] = f(pixels[i][j]);
            }
        }

    }
    else {

        vector <vector <int>> srt (flows, vector <int> (SZ));

        omp_set_num_threads(flows);
        #pragma omp parallel for schedule(KIND, CHUNK_SIZE) shared(srt)
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) srt[omp_get_thread_num()][pixels[i][j]]++;
        }
        
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

        omp_set_num_threads(flows);
        #pragma omp parallel for schedule(KIND, CHUNK_SIZE) shared(pixels)
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

void plus_P6 (FILE* fin, FILE* fout, double k) {

    int h, w, max_color;
    vector <vector <RGB>> pixels;

    read_P6 (fin, fout, h, w, max_color, pixels);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    int sz = 3 * h * w;
    unsigned char mn = 255;
    unsigned char mx = 0;

    auto f = [&] (unsigned char x) {
        int pxl = (int(x - mn) * SZ / int(mx - mn + 1));
        if (pxl < 0) return (unsigned char) 0;
        if (pxl > 255) return (unsigned char) 255;
        return (unsigned char) pxl;
    };

    if (flows == 0) {
        
        vector <int> srt (SZ);

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                srt[pixels[i][j].r]++;
                srt[pixels[i][j].g]++;
                srt[pixels[i][j].b]++;
            }
        }

        int pxls = 0;
        for (int i = 0; i <= 255; i++) {
            pxls += srt[i];
            if ((double) pxls > (double) sz * k) {
                mn = i;
                break;
            }
        }

        pxls = 0;
        for (int i = 255; i >= 0; i--) {
            pxls += srt[i];
            if ((double) pxls > (double) sz * k) {
                mx = i;
                break;
            }
        }

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                pixels[i][j].r = f(pixels[i][j].r);
                pixels[i][j].g = f(pixels[i][j].g);
                pixels[i][j].b = f(pixels[i][j].b);
            }
        }

    }
    else {

        vector <vector <int>> srt (flows, vector <int> (SZ));

        omp_set_num_threads(flows);
        #pragma omp parallel for schedule(KIND, CHUNK_SIZE) shared(srt)
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                srt[omp_get_thread_num()][pixels[i][j].r]++;
                srt[omp_get_thread_num()][pixels[i][j].g]++;
                srt[omp_get_thread_num()][pixels[i][j].b]++;
            }
        }

        
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

        

        omp_set_num_threads(flows);
        #pragma omp parallel for schedule(KIND, CHUNK_SIZE) shared(pixels)
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


int main (int argc, char const* argv[]) {

    if (argc != 4 && argc != 5) {
        cerr << "Error: Wrong number of arguments\n";
        exit(0);
    }

    flows = atoi(argv[1]);

    char const* fin_name = argv[2];
    char const* fout_name = argv[3];

    FILE* fin = fopen(fin_name, "rb");
    FILE* fout = fopen(fout_name, "wb");

    if (!fin) {
        cerr << "Error: Could not open the file\n";
        exit(0);
    }

    char type[2];
    fscanf(fin, "%c%c", &type[0], &type[1]);
    fprintf(fout, "%c%c ", type[0], type[1]);

    if (argc == 4) {
        if (type[0] = 'P' && type[1] == '5') treatment_P5(fin, fout);
        else if (type[0] = 'P' && type[1] == '6') treatment_P6(fin, fout);
        else {
            cerr << "Error: The file format is not supported\n";
            exit(0);
        }   
    }
        
    else {
        double k = stod (argv[4]);
        if (type[0] = 'P' && type[1] == '5') plus_P5(fin, fout, k);
        else if (type[0] = 'P' && type[1] == '6') plus_P6(fin, fout, k);
        else {
            cerr << "Error: The file format is not supported\n";
            exit(0);
        }   
    }

    fclose(fin);
    fclose(fout);

    return 0;
}