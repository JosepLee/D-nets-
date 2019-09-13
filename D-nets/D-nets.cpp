// D-nets.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
/*
 * Version 8.3
 * This program demonstrates D-Nets for image matching. The code implements
 * the exhaustive version (Clique D-Nets) from our CVPR2012 paper:
 *
 *            D-Nets: Beyond Patch-Based Image Descriptors
 *
 *            Felix von Hundelshausen       Rahul Sukthankar
 *            felix.v.hundelshausen@live.de rahuls@cs.cmu.edu
 *
 * IEEE International Conference on Computer Vision and Pattern Recognition
 *               June 18-20, 2012, Providence, Rhode Island, USA
 *
 * The program matches two images using FAST interest points as nodes.
 *
 * Copyright 2012, Felix von Hundelshausen and Rahul Sukthankar.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, is permitted provided that the following
 * conditions are met:
 *      Redistributions of source code must retain the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer.
 *      Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials
 *      provided with the distribution.
 *      The name of Contributors may not be used to endorse or
 *      promote products derived from this software without
 *      specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 * TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 * For visualizing the results this version uses a modified version of
 * original triangulation code by Steve Fortune. The disclaimer for
 * that respective part of our software is:
 *
 * "The author of this software is Steven Fortune.
 * Copyright (c) 1994 by AT&T Bell Laboratories.
 * Permission to use, copy, modify, and distribute this software for any
 * purpose without fee is hereby granted, provided that this entire notice
 * is included in all copies of any software which is or includes a copy
 * or modification of this software and in all copies of the supporting
 * documentation for such software.
 * THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTY.  IN PARTICULAR, NEITHER THE AUTHORS NOR AT&T MAKE ANY
 * REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY
 * OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE."
 */

 /*
 D-nets是一种基于条带的特征点匹配算法，在老师的项目中目的是得到更加精确的特征点对，
 在最开始D-nets替换sift得到的特征点时，某些图片得到了更高的匹配精确度，而对于后来
 的直线匹配算法，D-nets只是用来得到少量精确的特征点，也并非不可代替，用其他方法能
 得到同样的精确特征点对也可以，D-nets的使用非常复杂，需要在控制台中运行，运行参数
 不同，得到的效果也不同,main函数在3942行，大多数改动和注释都在main函数中
 */

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/legacy/legacy.hpp"

#include<fstream>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <set>

using namespace std;
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif
/*****************************************************************************/
float q0 = 0.1f;
float q1 = 0.8f;
int nSections = 9;
int bitsPerSection = 2;
float sigma = 1;
int nLayers = 8;
int pyramidAccessFactor = 8;
int nL = 20;
bool bWait = true;
int qualityMode = 1;
bool bDiscardMultipleHits = true;
int nExtractOnlyNBest = -1;
string outputFilename_VisualizedMatches;
int g_nChunkSize = 10;
bool g_bChunkedCreation = true;

// Variables for specific node extractors...
// FAST...
int FAST_thresh = 80;

// DENSE SAMPLING...
int g_dense_spacing = 10;
bool g_dense_bAddGaussianNoise = true;
float g_dense_stdDev = 3.0f;

// Variables that are initialized in the main() function
int nMaxDTokens;        // Total number of possible, distinct d-tokens
int nValuesPerSubSection;
float floatNValuesPerSubSection;
float g_geometricVerification_threshDist = 16;
typedef enum { FAST, SIFT, DENSE_SAMPLING } KeypointExtractionType;
KeypointExtractionType gKeypointExtractionType = FAST;
typedef enum { NONE, AFFINE, HOMOGRAPHY } TransformationModelType;
TransformationModelType gTransformationModel = HOMOGRAPHY;
typedef enum { LINES, MESHES } VisualizationType;
VisualizationType gVisualizationType = LINES;

// Visualization...
cv::Scalar color1 = cv::Scalar(0, 128, 255);
cv::Scalar color2 = cv::Scalar(0, 255, 0);
// Use cv::Scalar::all(-1) below if you prefer random colors.
cv::Scalar g_MatchColor = cv::Scalar(0, 0xff, 0xff);
cv::Scalar g_singlePointColor1 = cv::Scalar(0, 128, 255);
cv::Scalar g_singlePointColor2 = cv::Scalar(0, 255, 0);
int g_displayMeshLineWidth = 2;

/*****************************************************************************/
template <class REAL> inline REAL rnd() {
	return (REAL)rand() / (REAL)RAND_MAX;
}

/*****************************************************************************/
template <class REAL> REAL rndNormal(REAL mean, REAL var) {
	int l = 0;
	REAL s, v1;
	do {
		REAL u1 = rnd<REAL>();
		REAL u2 = rnd<REAL>();
		v1 = 2 * u1 - 1;
		REAL v2 = 2 * u2 - 1;
		s = v1 * v1 + v2 * v2;
		if (s < 1) { l = 1; }
	} while (!l);
	REAL z = sqrt(-2 * log(s) / s)*v1;
	REAL x = z * sqrt(var) + mean;
	return x;
}

template <class DataType> class fx_set : public std::set<DataType> {
public:
	bool contains(DataType d) {
		return std::find(this->begin(), this->end(), d) != this->end();
	}
	void remove(DataType d) {
		erase(std::find(this->begin(), this->end(), d));
	}
};

/*BEGIN DELAUNEY*/
////////////////////////////////////////
#define DELETED -2
#define le 0
#define re 1
////////////////////////////////////////
struct Freenode {
	struct Freenode *nextfree;
};
////////////////////////////////////////
struct Freelist {
	std::vector<char*> blocks;
	struct Freenode *head;
	int nodesize;
};
////////////////////////////////////////
struct Point {
	float x, y;
};
////////////////////////////////////////
struct Site { // structure used both for sites and for vertices
	struct Point	coord;
	int sitenbr;
	int refcnt;
};
////////////////////////////////////////
struct Edge {
	float a, b, c;
	struct Site *ep[2];
	struct Site *reg[2];
	int edgenbr;
};
////////////////////////////////////////
struct Halfedge {
	struct Halfedge *ELleft, *ELright;
	struct Edge *ELedge;
	int ELrefcnt;
	char ELpm;
	struct Site *vertex;
	float ystar;
	struct Halfedge *PQnext;
};
////////////////////////////////////////
struct I2 {
	int ia;
	int ib;
};
////////////////////////////////////////
struct I3 {
	int ia;
	int ib;
	int ic;
};
////////////////////////////////////////
class R2 {
public:
	R2() {}
	R2(int ia, int ib) :ia(ia), ib(ib) {}
	bool operator==(const R2&b) const {
		return ia == b.ia && ib == b.ib;
	}
	bool operator<(const R2&b) const {
		if (ia == b.ia) { return ib < b.ib; }
		return ia < b.ia;
	}
	int ia;
	int ib;
};
/*****************************************************************************/
//internal function
template <class A, class B>
int divide2arrays(A* a, B* b, int left, int right,
	bool(*a1_lessorequal_a2)(A& a1, A& a2)) {
	int i = left;
	int j = right - 1;
	A pivot = a[right];
	do {
		while (a1_lessorequal_a2(a[i], pivot) && i < right) { i++; }
		while (!a1_lessorequal_a2(a[j], pivot) && j > left) { j--; }
		if (i < j) {
			swap(a[i], a[j]);
			swap(b[i], b[j]);
		}
	} while (i < j);
	if (a1_lessorequal_a2(pivot, a[i])) {
		swap(a[i], a[right]);
		swap(b[i], b[right]);
	}
	return i;
}
/*****************************************************************************/
template <class A, class B>
void sort2ArraysInternal(A* a, B* b, int left, int right,
	bool(*a1_lessorequal_a2)(A& a1, A& a2)) {
	while (right > left) {
		int iDivide = divide2arrays(a, b, left, right, a1_lessorequal_a2);
		if (right - iDivide > iDivide - left) {
			sort2ArraysInternal(a, b, left, iDivide - 1, a1_lessorequal_a2);
			left = iDivide + 1;
		}
		else {
			sort2ArraysInternal(a, b, iDivide + 1, right, a1_lessorequal_a2);
			right = iDivide - 1;
		}
	}
}
/*****************************************************************************/
/* sorts two arrays of the same size simultaneously where the sorting is
 * induced by the first array.
 * Assume that you have two arrays
 * A a[n];
 * B b[n];
 * both having size n.
 * Assume further that you want to sort A based on a function that is
 * able to compare to elements of a (a1_lessorequal_a2).
 * The function sorts a according to this function.
 * but at the same time permutates the elements in array b, accordingly.
*/
template <class A, class B>
void sort2Arrays(A* a, B* b, int n,
	bool(*a1_lessorequal_a2)(A& a1, A& a2)) {
	sort2ArraysInternal(a, b, 0, n - 1, a1_lessorequal_a2);
}
//
/*****************************************************************************/
bool factorial(int& result, int n) {
	if (n < 0) { return false; }
	if (n == 0) {
		result = 1;
		return true;
	}
	if (n > 12) {
		printf("result would exceed interger limit\n");
		return false;
	}
	int m = n;
	result = 1;
	for (int i = 0; i < n - 1; i++) {
		result *= m;
		m--;
	}
	return true;
}
/*****************************************************************************/
/*!
   //see http://compprog.wordpress.com/2007/10/08/generating-permutations-2/
	Generates the next permutation of the vector v of length n.

	@return true, if there are no more permutations to be generated
	@return false, otherwise
*/
bool permutsDone(int v[], int n) {
	/* P2 */
	/* Find the largest i */
	int t;
	int i = n - 2;
	while ((i >= 0) && (v[i] > v[i + 1])) { --i; }
	/* If i is smaller than 0, then there are no more permutations. */
	if (i < 0) { return true; }
	/* Find the largest element after vi but not larger than vi */
	int k = n - 1;
	while (v[i] > v[k]) { --k; }
	t = v[i]; v[i] = v[k]; v[k] = t;

	/* Swap the last n - i elements. */
	int j;
	k = 0;
	for (j = i + 1; j < (n + i) / 2 + 1; ++j, ++k) {
		t = v[j]; v[j] = v[n - k - 1]; v[n - k - 1] = t;
	}

	return false;
}

/*****************************************************************************/
bool draw_k_from_n(int&result, int k, int n) {
	if (n < 0) { return false; }
	if (n == 0) {
		result = 0;
		return true;
	}
	if (!k) {
		result = 1;
		return true;
	}
	if (k<0 || k>n) {
		result = 0;
		return true;
	}
	int denom = 1;
	int mul = n;
	for (int i = 0; i < k; i++) {
		denom *= mul;
		mul--;
	}
	int fac_k;
	if (!factorial(fac_k, k)) { return false; }
	result = denom / fac_k;
	return true;
}
/*****************************************************************************/
int* create_draw_index_set(int& nIndexSets, int k, int outof_n) {
	if (k == outof_n) {
		int* x_1 = new int[k];
		for (int i = 0; i < k; i++) { x_1[i] = i; }
		nIndexSets = 1;
		return x_1;
	}
	if (!draw_k_from_n(nIndexSets, k, outof_n)) { return NULL; }
	if (nIndexSets == 0) { return NULL; }
	int* maxOfPos = new int[k];
	int* x = new int[nIndexSets*k];
#ifndef NDEBUG
	memset(x, 0xff, sizeof(int)*nIndexSets*k);//just for debugging, to see the memory operations better
#endif
	int* v = x;
	//setup the first row and initialize some variables
	for (int i = 0; i < k; i++) {
		maxOfPos[i] = outof_n - k + i;
		v[i] = i;
	}

	int j = 1;
	int* vn;
	int curpos = k - 1;
	do {
		vn = v + k;
		// copy the last line, at the same time detect the first value
		// that has reached its maximum
		int i = 0;
		bool bMaxReached = false;
		do {
			int a = v[i];
			if (a == maxOfPos[i]) {
				bMaxReached = true;
				break;
			}
			vn[i] = a;
			i++;
		} while (i < k);
		if (bMaxReached) {
			//line feed
			int b = v[i - 1] + 1;
			vn[i - 1] = b;
			for (int r = i; r < k; r++) {
				vn[r] = ++b;
			}
			curpos = k - 1;
		}
		else {
			vn[curpos]++;
		}
		j++;
		v = vn;
	} while (j < nIndexSets);

	delete[] maxOfPos;
	return x;
}

/*****************************************************************************/
template <class REAL> inline REAL TAbs(REAL a) {
	return (a < 0.0) ? -a : a;
};
/*****************************************************************************/
template <class REAL> inline bool exactly_equal(REAL a, REAL b) {
	return !(a < b) && !(b < a);
}
/*****************************************************************************/
template<class REAL> void swapvalues(REAL&a, REAL&b)
{
	REAL c = a;
	a = b;
	b = c;
}
/*****************************************************************************/
template <class REAL> inline REAL wrapAngle(REAL phiFrom, REAL phiTo, REAL len)
{
	REAL a = phiTo - phiFrom;
	REAL b = len - a;
	if (TAbs(a) <= TAbs(b)) {
		return a;
	}
	else {
		return -b;
	}
}
/*****************************************************************************/
template <class REAL> inline void NormalizePhi(REAL& phi) {
	if (phi < 0.0) { phi += ((REAL)2.0*(REAL)M_PI); }
	if (phi > ((REAL)2.0*(REAL)M_PI)) { phi -= ((REAL)2.0*(REAL)M_PI); }
}
/*****************************************************************************/
template <class REAL> inline REAL angleFromTo(REAL phiFrom, REAL phiTo)
{
	NormalizePhi(phiFrom);
	NormalizePhi(phiTo);
	REAL sig = 1;
	if (phiFrom > phiTo) {
		sig = -1;
		swapvalues(phiFrom, phiTo);
	}
	REAL dphi = wrapAngle(phiFrom, phiTo, ((REAL)2.0*(REAL)M_PI));
	return dphi * sig;
}
/*****************************************************************************/

std::vector< pair<int, char*> > imageTypeStringMapping;
cv::Mat markerMask, img;
cv::Point prevPt(-1, -1);

/*****************************************************************************/
struct IndexMapInfo {
	int index;
	int i1stBestMatch;
	int i2ndBestMatch;
	float p1;
	float p2;
	float power;
	float entropy;
	float quality;
};

/*****************************************************************************/
int IndexMapInfo_Quality_Cmp(const void*a, const void*b) {
	IndexMapInfo* s1 = (struct IndexMapInfo*) a;
	IndexMapInfo* s2 = (struct IndexMapInfo*) b;
	if (s1->quality < s2->quality) { return 1; }
	if (s1->quality > s2->quality) { return -1; }
	return(0);
}

/*****************************************************************************/
template <class REAL>
REAL sumVector(REAL* v, int n) {
	REAL sum = 0;
	for (int i = 0; i < n; i++) { sum += v[i]; }
	return sum;
}

/*****************************************************************************/
template <class REAL>
bool getMinAndMaxOfVector(REAL& min, REAL& max, REAL* v, int n) {
	min = FLT_MAX;
	max = -FLT_MAX;
	for (int i = 0; i < n; i++) {
		REAL a = v[i];
		if (a < min) { min = a; }
		if (a > max) { max = a; }
	}
	return n > 0;
}

/*****************************************************************************/
template <class REAL> bool normalizeVectorMinMaxZeroOne(REAL* v, int n) {
	REAL min, max;
	if (!getMinAndMaxOfVector(min, max, v, n)) { return false; }
	REAL d = max - min;
	if (d == 0) { return false; }
	REAL factor = (REAL)1.0 / d;
	for (int i = 0; i < n; i++) {
		REAL a = v[i];
		v[i] = (a - min)*factor;
	}
	return true;
}

/*****************************************************************************/
// Use this only if you know that all the elements of v are non-negative (>= 0)
template <class REAL> REAL normalizeVector_noabs(REAL* v, int n) {
	REAL sum = sumVector(v, n);
	if (sum == 0) { return sum; }
	REAL fac = ((REAL)1) / sum;
	for (int i = 0; i < n; i++) {
		v[i] *= fac;
	}
	return sum;
}
/*****************************************************************************/
const char* depthCode2String(int depth) {
	for (unsigned int i = 0; i < imageTypeStringMapping.size(); i++) {
		if (imageTypeStringMapping[i].first == depth) {
			return imageTypeStringMapping[i].second;
		}
	}
	return "unknown depth code";
}

/*****************************************************************************/
bool ensureGrayImageWithDepth(cv::Mat& imgGray,
	int target_depth,
	cv::Mat& img,
	char*image_name = NULL,
	bool bPrintFormatInfo = true) {
	cv::Size sz = img.size();
	int nC = img.channels();
	int depth = img.depth();
	if (bPrintFormatInfo) {
		cout << "input image \"" << (image_name ? image_name : "") << "\" has "
			<< nC << " channels of format " << depthCode2String(depth)
			<< " with " << sz.width << "x" << sz.height << " pixels.\n";
	}
	cv::Mat grayButWithSourceDepth;
	switch (nC) {
	case 1: img.copyTo(grayButWithSourceDepth);
		break;
	case 3: switch (depth) {
	case CV_8U: case CV_16U: case CV_32F:
		cvtColor(img, grayButWithSourceDepth, CV_BGR2GRAY);
		break;
	default:
		return false;
	}
			break;
	case 4: switch (depth) {
	case CV_8U: case CV_16U: case CV_32F:
		cvtColor(img, grayButWithSourceDepth, CV_BGRA2GRAY);
		break;
	default:
		return false;
	}
			break;
	default:
		return false;
	}
	int inter_nC = grayButWithSourceDepth.channels();
	int inter_depth = grayButWithSourceDepth.depth();
	if (inter_nC != 1) {
		cout << "only one channel expected after first step of conversion\n";
		return false;
	}
	switch (target_depth) {
	case CV_8U:
		switch (inter_depth) {
		case CV_8U:
			grayButWithSourceDepth.copyTo(imgGray);
			break;
		case CV_16U:
			grayButWithSourceDepth.convertTo(imgGray, CV_8U, 255.0 / 65535.0);
			break;
		case CV_32F:
			grayButWithSourceDepth.convertTo(imgGray, CV_8U, 255.0);
			break;
		}
		break;
	case CV_32F:
		switch (inter_depth) {
		case CV_8U:
			grayButWithSourceDepth.convertTo(imgGray, CV_32F, 1.0 / 255.0);
			break;
		case CV_16U:
			grayButWithSourceDepth.convertTo(imgGray, CV_32F, 1.0 / 65535.0);
			break;
		case CV_32F:
			grayButWithSourceDepth.copyTo(imgGray);
			break;
		}
		break;
	}
	int final_nC = imgGray.channels();
	int final_depth = imgGray.depth();
	cv::Size final_sz = imgGray.size();
	if (final_sz != sz || final_nC != 1 || final_depth != target_depth) {
		return false;
	}
	return true;
}

/*****************************************************************************/
bool readParam(string&flag, string&param, bool&bFloatValid, float&value,
	char*arg) {
	// remove optional preceeding '-'
	while (*arg == '-') { arg++; }
	int l = strlen(arg);
	if (l < 3) { return false; }
	// search for '='
	int ie = 1;
	while (ie < l && arg[ie] != '=') { ie++; }
	if (ie == l) { return false; }  // Failed to find '='
	flag = string(arg, ie);
	param = string(arg + ie + 1, l - ie);
	bFloatValid = (sscanf(param.c_str(), "%f", &value) == 1);
	return true;
}

/*****************************************************************************/
bool readAndConvertImages(cv::Mat imgGray[2], int argc, char** argv) {
	bool bFixedWidth[2] = { true,true };
	int fixedWidth[2];
	float scaleFactor[2];
	char* filename[2];
	char* resizeString[2];
	if (argc < 5) {
		printf("you need to specify two files, each having an additional scale spec parameter (run ./d-nets -h for help)");
		return false;
	}
	filename[0] = argv[1];
	resizeString[0] = argv[2];
	filename[1] = argv[3];
	resizeString[1] = argv[4];

	cv::Mat img[2];
	for (int i = 0; i < 2; i++) {
		img[i] = cv::imread(filename[i], 1);
		if (img[i].empty()) {
			cout << "Failed to open image \"" << filename[i] << "\".\n";
			return false;
		}
		if (!ensureGrayImageWithDepth(imgGray[i], CV_32F, img[i], filename[i], true)) {
			cout << "Failed to convert image \"" << filename[i]
				<< "\" to a gray image with float values.\n";
			return false;
		}

		//resize images...
		string flag;
		float value;
		bool bValueOk;
		string param;
		if (readParam(flag, param, bValueOk, value, resizeString[i]) &&
			(flag == "w" || flag == "s") && bValueOk) {
			if (flag == "w") {
				bFixedWidth[i] = true;
				fixedWidth[i] = (int)value;
			}
			else if (flag == "s") {
				bFixedWidth[i] = false;
				scaleFactor[i] = value;
			}
			cv::Size sz = imgGray[i].size();
			if (bFixedWidth[i]) {
				scaleFactor[i] = (float)fixedWidth[i] / (float)sz.width;
			}
			if (scaleFactor[i] != 1.) {
				cv::Mat tmp;
				cv::resize(imgGray[i], tmp, cv::Size(0, 0), scaleFactor[i], scaleFactor[i],
					CV_INTER_AREA);
				tmp.copyTo(imgGray[i]);
			}
		}
	}
	return true;
}

/*****************************************************************************/
void init() {
	imageTypeStringMapping.push_back(pair<int, char*>(CV_8U, (char*)"CV_8U"));
	imageTypeStringMapping.push_back(pair<int, char*>(CV_8S, (char*)"CV_8S"));
	imageTypeStringMapping.push_back(pair<int, char*>(CV_16U, (char*)"CV_16U"));
	imageTypeStringMapping.push_back(pair<int, char*>(CV_16S, (char*)"CV_16S"));
	imageTypeStringMapping.push_back(pair<int, char*>(CV_32S, (char*)"CV_32S"));
	imageTypeStringMapping.push_back(pair<int, char*>(CV_32F, (char*)"CV_32F"));
	imageTypeStringMapping.push_back(pair<int, char*>(CV_64F, (char*)"CV_64F"));
}
/*****************************************************************************/
template <class REAL> class TV2;
typedef class TV2<float> V2;
typedef class TV2<double> DV2;

template <class REAL> class TV2 {
public:
	TV2() { x = (REAL)0; y = (REAL)0; }
	TV2(REAL x, REAL y) : x(x), y(y) {};
	TV2(const V2& v) : x((REAL)v.x), y((REAL)v.y) {}
	TV2(const DV2& v) : x((REAL)v.x), y((REAL)v.y) {}
	inline TV2<REAL> operator-(const TV2<REAL> &v) const {
		return TV2(x - v.x, y - v.y);
	}
	inline REAL len() const { return (REAL)sqrt(x*x + y * y); }
	inline REAL operator!() {
		REAL p = atan2(this->y, this->x);
		if (p < 0) { p += 2.0*M_PI; }
		return p;
	}
	inline REAL angleTo(TV2 v) { return angleFromTo(!*this, !v); }
	inline REAL toOne() {
		REAL l = len();
		if (exactly_equal(l, (REAL)0)) { return 0; }
		x /= l;
		y /= l;
		return l;
	}
	inline TV2<REAL> asOne() {
		REAL l = len();
		if (exactly_equal(l, (REAL)0)) { return *this; }
		return TV2<REAL>(x / l, y / l);
	}
	inline REAL operator*(const TV2& v) const {
		return x * v.x + y * v.y;
	}
	inline TV2<REAL> operator~() const {
		return TV2((-1)*y, x);
	}
	inline TV2<REAL> operator+(const TV2<REAL> &v) const {
		return TV2(x + v.x, y + v.y);
	}
	REAL x;
	REAL y;
};

/*****************************************************************************/
struct TwoV2 {
	V2 a;
	V2 b;
};
/*****************************************************************************/
template <class REAL> class TV3
{
public:
	TV3() { x = y = z = 0; }
	TV3(REAL x, REAL y, REAL z) : x(x), y(y), z(z) {}
	TV3(const TV3& v) : x(v.x), y(v.y), z(v.z) {}
	TV3(const TV2<REAL>&s) : x(s.x), y(s.y), z(0) {}

	//inline TV3<REAL> operator-(const TV3&v) const;
	//inline TV3<REAL> operator+(const TV3&v) const;
	//inline TV3<REAL> operator*(REAL m) const;
	//inline TV3<REAL> operator/(REAL d) const;
	//inline TV3<REAL> operator-()const;
	inline REAL operator*(const TV3&v) const { return x * v.x + y * v.y + z * v.z; }
	//inline TV3<REAL> operator |(const TV3&v);
	//inline void operator /=(REAL f);
	inline void operator*=(REAL f) { x *= f; y *= f; z *= f; };
	//inline void operator +=(const TV3&v);
	//inline void operator -=(const TV3&v);
	//inline bool operator ==(const TV3&v) const;
	//inline bool operator !=(const TV3&v) const;
	//inline TV3<REAL>& operator=(const TV2<REAL> &v){x=v.x;y=v.y;z=0;return *this;};
	//inline operator TV2<REAL>() const{return TV2<REAL>(x,y);};
	REAL x, y, z;
};

typedef  class TV3<float> V3;
typedef  class TV3<double> DV3;
/*****************************************************************************/
template <class REAL> class TAT2 {
public:
	TAT2() {};
	//////////////////////////////////////////
	TAT2(const TAT2<REAL>& s) {
		a[0] = s.a[0];
		a[1] = s.a[1];
		a[2] = s.a[2];
		a[3] = s.a[3];
		t = s.t;
	};
	//////////////////////////////////////////
	~TAT2() {};
	//////////////////////////////////////////
	TAT2<REAL>& operator=(const TAT2<REAL>&s) {
		a[0] = s.a[0];
		a[1] = s.a[1];
		a[2] = s.a[2];
		a[3] = s.a[3];
		t = s.t;
		return *this;
	}
	//////////////////////////////////////////
	bool operator==(const TAT2<REAL>&other) const {
		if (t.x != other.t.x) { return false; }
		if (t.y != other.t.y) { return false; }
		if (a[0] != other.a[0]) { return false; }
		if (a[1] != other.a[1]) { return false; }
		if (a[2] != other.a[2]) { return false; }
		if (a[3] != other.a[3]) { return false; }
		return true;
	}
	//////////////////////////////////////////
	bool defineFromTriangles(TV2<REAL> triFrom[3], TV2<REAL> triTo[3]) {
		REAL ax = triFrom[0].x;
		REAL ay = triFrom[0].y;
		REAL bx = triFrom[1].x;
		REAL by = triFrom[1].y;
		REAL cx = triFrom[2].x;
		REAL cy = triFrom[2].y;
		REAL A = ax * (by - cy) + bx * (cy - ay) + cx * (ay - by);//TODO:optimize see a**
		if (A == 0) { return false; }
		REAL Ai = 1.0f / A;
		REAL axs = triTo[0].x;
		REAL ays = triTo[0].y;
		REAL bxs = triTo[1].x;
		REAL bys = triTo[1].y;
		REAL cxs = triTo[2].x;
		REAL cys = triTo[2].y;
		TV3<REAL> t_bary(bx*cy - cx * by, cx*ay - ax * cy, ax*by - ay * bx);
		TV3<REAL> Q1(by - cy, cy - ay, ay - by);//a**
		TV3<REAL> Q2(cx - bx, ax - cx, bx - ax);
		TV3<REAL> v1(axs, bxs, cxs);
		TV3<REAL> v2(ays, bys, cys);
		v1 *= Ai;
		v2 *= Ai;
		a[0] = v1 * Q1;
		a[1] = v1 * Q2;
		a[2] = v2 * Q1;
		a[3] = v2 * Q2;
		t.x = v1 * t_bary;
		t.y = v2 * t_bary;
		return true;
	}
	//////////////////////////////////////////
	bool defineFromTriangles(TV2<REAL>& from0,
		TV2<REAL>& from1,
		TV2<REAL>& from2,
		TV2<REAL>& to0,
		TV2<REAL>& to1,
		TV2<REAL>& to2) {
		REAL A = from0.x*(from1.y - from2.y) +
			from1.x*(from2.y - from0.y) +
			from2.x*(from0.y - from1.y);  //TODO:optimize see a**
		if (A == 0) { return false; }
		REAL Ai = 1.0f / A;
		TV3<REAL> t_bary(from1.x*from2.y - from2.x*from1.y,
			from2.x*from0.y - from0.x*from2.y,
			from0.x*from1.y - from0.y*from1.x);
		TV3<REAL> Q1(from1.y - from2.y, from2.y - from0.y, from0.y - from1.y);//a**
		TV3<REAL> Q2(from2.x - from1.x, from0.x - from2.x, from1.x - from0.x);
		TV3<REAL> v1(to0.x, to1.x, to2.x);
		TV3<REAL> v2(to0.y, to1.y, to2.y);
		v1 *= Ai;
		v2 *= Ai;
		a[0] = v1 * Q1;
		a[1] = v1 * Q2;
		a[2] = v2 * Q1;
		a[3] = v2 * Q2;
		REAL det = a[0] * a[3] - a[1] * a[2];
		if (det == 0) { return false; }
		t.x = v1 * t_bary;
		t.y = v2 * t_bary;
		return true;
	}
	//////////////////////////////////////////
	void defineAsIdentity() {
		a[0] = 1.0f;
		a[1] = 0.0f;
		a[2] = 0.0f;
		a[3] = 1.0f;
		t.x = 0;
		t.y = 0;
	}
	//////////////////////////////////////////
	TV2<REAL> out(TV2<REAL> p) {
		TV2<REAL> ps;
		ps.x = a[0] * p.x + a[1] * p.y + t.x;
		ps.y = a[2] * p.x + a[3] * p.y + t.y;
		return ps;
	}
	//////////////////////////////////////////
	TV2<REAL> vout(TV2<REAL> v) {
		TV2<REAL> ps;
		ps.x = a[0] * v.x + a[1] * v.y;
		ps.y = a[2] * v.x + a[3] * v.y;
		return ps;
	}
	//////////////////////////////////////////
	bool calcInverseTransformation(TAT2<REAL>& at) {
		REAL det = a[1] * a[2] - a[0] * a[3];
		if (det == 0) { return false; }
		at.a[0] = -a[3] / det;
		at.a[1] = a[1] / det;
		at.a[2] = a[2] / det;
		at.a[3] = -a[0] / det;
		at.t.x = -at.a[0] * t.x - at.a[1] * t.y;
		at.t.y = -at.a[2] * t.x - at.a[3] * t.y;
		return true;

	}
	//////////////////////////////////////////
	REAL phiOut(REAL phi) {
		TV2<REAL> v(cos(phi), sin(phi));
		TV2<REAL> vt = vout(v);
		float phit = !vt;
		return phit;
	}
	/*   //////////////////////////////////////////
	   TV2phi<REAL> v2phiOut(TV2phi<REAL> p)
	   {
		  TV2phi<REAL> v2phit;
		  TV2<REAL> pt=out(TV2<REAL>(p.x,p.y));
		  float phiout=phiOut(p.phi);
		  return TV2phi<REAL>(pt.x,pt.y,phiout);
	   }*/
	   //////////////////////////////////////////
	REAL calcError(TAT2<REAL>& c) {
		TV2<REAL> ref[3];
		ref[0].x = 0.0f;
		ref[0].y = 0.0f;
		ref[1].x = 1.0f;
		ref[1].y = 0.0f;
		ref[2].x = 0.0f;
		ref[2].y = 1.0f;
		TV2<REAL> t1[3];
		TV2<REAL> t2[3];
		REAL sumDist = 0;
		for (int i = 0; i < 3; i++) {
			t1[i] = out(ref[i]);
			t2[i] = c.out(ref[i]);
			TV2<REAL> v = t1[i] - t2[i];
			REAL l = v.len();
			sumDist += l;
		}
		return sumDist;
	}
	//////////////////////////////////////////
	void defineAsRigidTransformation(REAL phi, TV2<REAL> nt) {
		t.x = nt.x;
		t.y = nt.y;
		REAL c = cos(phi);
		REAL s = sin(phi);
		a[0] = c;
		a[2] = s;
		a[1] = -s;
		a[3] = c;
	}
	//////////////////////////////////////////
	void defineAsScaledRigidTransformation(REAL scale, REAL phi, TV2<REAL> nt) {
		t.x = nt.x;
		t.y = nt.y;
		REAL c = scale * cosf(phi);
		REAL s = scale * sinf(phi);
		a[0] = c;
		a[2] = s;
		a[1] = -s;
		a[3] = c;
	}
	//////////////////////////////////////////
	bool defineAsScaledRigidTransformation(TV2<REAL> fromA,
		TV2<REAL> fromB,
		TV2<REAL> toA,
		TV2<REAL> toB) {
		TV2<REAL> vSource = fromB - fromA;
		TV2<REAL> vTarget = toB - toA;
		REAL lSource = vSource.len();
		REAL lTarget = vTarget.len();
		if (lSource == 0) { return false; }
		REAL scale = lTarget / lSource;
		defineAsScaledRigidTransformation(scale,
			angleFromTo(!vSource, !vTarget),
			TV2<REAL>());
		t = toA - out(fromA);
		return true;
	}
	//////////////////////////////////////////
   /* bool defineAsRigidTransformationAround(TV2<REAL> center, REAL dphi, TV2<REAL> dt)
	{
	   TV2<REAL> p[3],ps[3];
	   p[0]=TV2<REAL>(0.0f,0.0f);
	   p[1]=TV2<REAL>(1.0f,0.0f);
	   p[2]=TV2<REAL>(0.0f,1.0f);
	   for (int i=0;i<3;i++){
		  TV2<REAL> v=p[i]-center;
		  v.rotate(dphi);
		  ps[i]=center+v+dt;
	   }
	   return defineFromTriangles(p,ps);
	}*/
	//////////////////////////////////////////
	// Return the absolute delta angle in [rad] by which the
	// absolute angle between the column vectors of the a matrix differs
	// from 90 degrees.
	REAL getAngularDeviationFromBeingOrthogonal() {
		TV2<REAL> u(a[0], a[2]);
		TV2<REAL> v(a[1], a[3]);
		REAL dphi_from_ortho = fabs(angleFromTo((REAL)(M_PI / 2.0),
			(REAL)fabs(u.angleTo(v))));
		return dphi_from_ortho;
	}
	//////////////////////////////////////////
	REAL getRigidity() {
		TV2<REAL> va(a[0], a[2]);
		TV2<REAL> vb(a[1], a[3]);
		va.toOne();
		vb.toOne();
		REAL rigidity = va * vb;
		return 1.0 - fabs(rigidity);
	}
	//////////////////////////////////////////
	void getLengthOfAxes(REAL&lu, REAL&lv) {
		TV2<REAL> u(a[0], a[2]);
		TV2<REAL> v(a[1], a[3]);
		lu = u.len();
		lv = v.len();
	}
	//////////////////////////////////////////
	REAL a[4];
	TV2<REAL> t;
};
//////////////////////////////////////////////////////////////////////////////
typedef class TAT2<float> AT2;
typedef class TAT2<double> DAT2;  //(D)ouble (A)ffine (T)ransformation (2)d
////////////////////////////////////////
//The following class Delauney2 uses/modifies code of Steve Fortune's.
//The original disclaimer for that code is:
/*
 * The author of this software is Steven Fortune.  Copyright (c) 1994 by AT&T
 * Bell Laboratories.
 * Permission to use, copy, modify, and distribute this software for any
 * purpose without fee is hereby granted, provided that this entire notice
 * is included in all copies of any software which is or includes a copy
 * or modification of this software and in all copies of the supporting
 * documentation for such software.
 * THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTY.  IN PARTICULAR, NEITHER THE AUTHORS NOR AT&T MAKE ANY
 * REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY
 * OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
 */
class Triangulator {
public:
	Triangulator(bool bOutEdges, bool bOutTriangles, V2*p, int nP,
		int triangulate = 1, int sorted = 0, int plot = 0, int debug = 0);
	virtual ~Triangulator();
	void freeinit(struct	Freelist *fl, int size);
	char *getfree(struct	Freelist *fl);
	void makefree(struct Freenode *curr, struct Freelist *fl);
	void releaseList(struct Freelist*fl);
	char *myalloc(unsigned n);
	void geominit();
	void plotinit();
	void iniSites(V2*p, int nP);
	struct Site *next();
	void PQinitialize();
	void PQinsert(struct Halfedge *he, struct Site *v, float 	offset);
	void PQdelete(struct Halfedge *he);
	int PQbucket(struct Halfedge *he);
	struct Halfedge *PQextractmin();

	int PQempty();
	struct Point PQ_min();
	void ELinitialize();
	void ELinsert(struct	Halfedge *lb, struct	Halfedge * newone);
	void ELdelete(struct Halfedge *he);
	struct Halfedge * ELright(struct Halfedge *he);
	struct Halfedge * ELleft(struct Halfedge *he);
	struct Halfedge * ELgethash(int b);
	struct Halfedge * ELleftbnd(struct Point *p);

	struct Halfedge *HEcreate(struct Edge *e, int pm);
	void out_site(struct Site *s);
	void out_vertex(struct Site *v);
	void out_bisector(struct Edge *e);
	void out_ep(struct Edge *e);
	void out_triple(struct Site *s1, struct Site *s2, struct Site * s3);
	void clip_line(struct Edge *e);
	void line(int x1, int y1, int x2, int y2);
	int right_of(struct Halfedge *el, struct Point *p);
	struct Site *leftreg(struct Halfedge *he);
	struct Site *rightreg(struct Halfedge *he);
	struct Edge *bisect(struct	Site *s1, struct Site*s2);
	struct Site *intersect(struct Halfedge *el1, struct Halfedge *el2, struct Point *p = NULL);
	float dist(struct Site *s, struct Site *t);
	void deref(struct	Site *v);
	void ref(struct Site *v);
	void makevertex(struct Site *v);
	void endpoint(struct Edge *e, int lr, struct Site *s);

	void voronoi();

	struct Site *sites;
	int nsites;
	int siteidx;
	int sqrt_nsites;
	int nvertices;
	struct Freelist sfl;
	struct Site *bottomsite;
	int total_alloc;
	int triangulate;
	int sorted;
	int plot;
	int debug;
	float xmin, xmax, ymin, ymax, deltax, deltay;
	struct Halfedge *PQhash;
	int PQhashsize;
	int PQcount;
	int PQmin;
	int ntry;
	int totalsearch;
	struct Freelist hfl;
	struct Halfedge *ELleftend, *ELrightend;
	int 	ELhashsize;
	struct Halfedge **ELhash;
	float pxmin, pxmax, pymin, pymax, cradius;
	int nedges;
	struct Freelist efl;
	std::vector<I3> final_triangles;
	fx_set<R2> final_edges;
	bool bOutEdges;
	bool bOutTriangles;
	int nMallocs;//for debugging
};
/////////////////////////////////////////////////
Triangulator::Triangulator(bool bOutEdges, bool bOutTriangles, V2* p, int nP,
	int triangulate, int sorted, int plot, int debug)
	: triangulate(triangulate), sorted(sorted), plot(plot),
	debug(debug), bOutEdges(bOutEdges), bOutTriangles(bOutTriangles) {
	nMallocs = 0;
	PQhash = NULL;
	ELhash = NULL;
	total_alloc = 0;
	sites = NULL;
	freeinit(&sfl, sizeof(*sites));
	iniSites(p, nP);
	siteidx = 0;
	geominit();
	if (plot) { plotinit(); }
	voronoi();
}
/////////////////////////////////////////////////
Triangulator::~Triangulator() {
	if (sites) {
		free(sites);
		sites = NULL;
	}
	if (ELhash) {
		free(ELhash);
		ELhash = NULL;
	}
	if (PQhash) {
		free(PQhash);
		PQhash = NULL;
	}
	releaseList(&sfl);
	releaseList(&efl);
	releaseList(&hfl);
}
/////////////////////////////////////////////////
void Triangulator::freeinit(struct	Freelist *fl, int size) {
	fl->head = (struct Freenode *) NULL;
	fl->nodesize = size;
}
/////////////////////////////////////////////////
void Triangulator::releaseList(struct Freelist*fl) {
	for (int i = 0; i < fl->blocks.size(); i++) {
		free(fl->blocks[i]);
	}
	fl->blocks.clear();
	fl->head = NULL;
	fl->nodesize = 0;
}
/////////////////////////////////////////////////
char *Triangulator::getfree(struct	Freelist *fl)
{
	int i;
	struct Freenode *t;
	if (fl->head == (struct Freenode *) NULL) {
		t = (struct Freenode *) myalloc(sqrt_nsites * fl->nodesize);
		fl->blocks.push_back((char*)t);
		for (i = 0; i < sqrt_nsites; i++) {
			makefree((struct Freenode *)((char *)t + i * fl->nodesize), fl);
		}
	};
	t = fl->head;
	fl->head = fl->head->nextfree;
	return (char *)t;
}
/////////////////////////////////////////////////
void Triangulator::makefree(struct Freenode *curr, struct Freelist *fl) {
	curr->nextfree = fl->head;
	fl->head = curr;
}
/////////////////////////////////////////////////
char *Triangulator::myalloc(unsigned n) {
	char *t = (char*)malloc(n);
	if (!t) {
		fprintf(stderr, "Insufficient memory site %d (%d bytes in use)\n",
			siteidx, total_alloc);
		return NULL;
	};
	nMallocs++;
	total_alloc += n;
	return t;
}
/////////////////////////////////////////////////
void Triangulator::geominit() {
	struct Edge e;
	float sn;
	freeinit(&efl, sizeof e);
	nvertices = 0;
	nedges = 0;
	sn = nsites + 4;
	sqrt_nsites = sqrt(sn);
	deltay = ymax - ymin;
	deltax = xmax - xmin;
}
/////////////////////////////////////////////////
void Triangulator::plotinit() {
	float dx, dy, d;

	dy = ymax - ymin;
	dx = xmax - xmin;
	d = (dx > dy ? dx : dy) * 1.1;
	pxmin = xmin - (d - dx) / 2.0;
	pxmax = xmax + (d - dx) / 2.0;
	pymin = ymin - (d - dy) / 2.0;
	pymax = ymax + (d - dy) / 2.0;
	cradius = (pxmax - pxmin) / 350.0;
	//openpl();
	//range(pxmin, pymin, pxmax, pymax);
}
/////////////////////////////////////////////////
/* sort sites on y, then x, coord */
int scomp(const void*a, const void*b) {
	struct Point*s1 = (struct Point*)a;
	struct Point*s2 = (struct Point*)b;
	if (s1->y < s2->y) { return(-1); }
	if (s1->y > s2->y) { return(1); }
	if (s1->x < s2->x) { return(-1); }
	if (s1->x > s2->x) { return(1); }
	return 0;
}
////////////////////////////////////////////////////////////////////////////////
void Triangulator::iniSites(V2*p, int nP) {
	sites = (struct Site*) myalloc(nP * sizeof(*sites));
	for (int i = 0; i < nP; i++) {
		sites[i].coord.x = p[i].x;
		sites[i].coord.y = p[i].y;
		sites[i].sitenbr = i;
		sites[i].refcnt = 0;
	}
	nsites = nP;

	qsort(sites, nsites, sizeof(*sites), scomp);
	xmin = sites[0].coord.x;
	xmax = sites[0].coord.x;
	for (int i = 1; i < nsites; i += 1) {
		if (sites[i].coord.x < xmin) { xmin = sites[i].coord.x; }
		if (sites[i].coord.x > xmax) { xmax = sites[i].coord.x; }
	}
	ymin = sites[0].coord.y;
	ymax = sites[nsites - 1].coord.y;
}
/////////////////////////////////////////////////
struct Site *Triangulator::next() {
	struct Site *s;
	if (siteidx < nsites) {
		s = &sites[siteidx];
		siteidx += 1;
		return s;
	}
	else {
		return (struct Site *)NULL;
	}
}
/////////////////////////////////////////////////
void Triangulator::PQinitialize() {
	int i;
	struct Point *s;
	PQcount = 0;
	PQmin = 0;
	PQhashsize = 4 * sqrt_nsites;
	PQhash = (struct Halfedge *) myalloc(PQhashsize * sizeof *PQhash);
	for (i = 0; i < PQhashsize; i++) {
		PQhash[i].PQnext = (struct Halfedge *)NULL;
	}
}
/////////////////////////////////////////////////
struct Halfedge *Triangulator::PQextractmin() {
	struct Halfedge *curr;
	curr = PQhash[PQmin].PQnext;
	PQhash[PQmin].PQnext = curr->PQnext;
	PQcount--;
	return curr;
}
/////////////////////////////////////////////////
void Triangulator::out_site(struct Site *s) {
	if (!triangulate & plot & !debug) {
		//circle (s->coord.x, s->coord.y, cradius);
		printf("circle visualization not implemented\n");
	}
	if (!triangulate & !plot & !debug) {
		printf("s %f %f\n", s->coord.x, s->coord.y);
	}
	if (debug) {
		printf("site (%d) at %f %f\n", s->sitenbr, s->coord.x, s->coord.y);
	}
}
/////////////////////////////////////////////////
void Triangulator::out_vertex(struct Site *v) {
	if (!triangulate & !plot & !debug) {
		printf("v %f %f\n", v->coord.x, v->coord.y);
	}
	if (debug) {
		printf("vertex(%d) at %f %f\n", v->sitenbr, v->coord.x, v->coord.y);
	}
}
/////////////////////////////////////////////////
void Triangulator::out_bisector(struct Edge *e)
{
	if (triangulate & plot & !debug) {
		line(e->reg[0]->coord.x, e->reg[0]->coord.y,
			e->reg[1]->coord.x, e->reg[1]->coord.y);
	}
	if (!triangulate & !plot & !debug) {
		printf("l %f %f %f", e->a, e->b, e->c);
	}
	if (debug) {
		printf("line(%d) %gx+%gy=%g, bisecting %d %d\n",
			e->edgenbr, e->a, e->b, e->c,
			e->reg[le]->sitenbr, e->reg[re]->sitenbr);
	}
}
/////////////////////////////////////////////////
void Triangulator::out_triple(struct Site *s1, struct Site *s2, struct Site * s3) {
	if (bOutTriangles) {
		I3 i3;
		i3.ia = s1->sitenbr;
		i3.ib = s2->sitenbr;
		i3.ic = s3->sitenbr;
		final_triangles.push_back(i3);
	}
	if (bOutEdges) {
		R2 e1(s1->sitenbr, s2->sitenbr);
		R2 e2(s2->sitenbr, s3->sitenbr);
		R2 e3(s3->sitenbr, s1->sitenbr);
		R2 e1m(s2->sitenbr, s1->sitenbr);
		R2 e2m(s3->sitenbr, s2->sitenbr);
		R2 e3m(s1->sitenbr, s3->sitenbr);
		if (!final_edges.contains(e1) && !final_edges.contains(e1m)) {
			final_edges.insert(e1);
		}
		if (!final_edges.contains(e2) && !final_edges.contains(e2m)) {
			final_edges.insert(e2);
		}
		if (!final_edges.contains(e3) && !final_edges.contains(e3m)) {
			final_edges.insert(e3);
		}
	}
	/*
	if(triangulate & !plot &!debug)
		printf("%d %d %d\n", s1->sitenbr, s2->sitenbr, s3->sitenbr);
	if(debug)
		printf("circle through left=%d right=%d bottom=%d\n",
			s1->sitenbr, s2->sitenbr, s3->sitenbr);
		  */
}
/////////////////////////////////////////////////
void Triangulator::out_ep(struct Edge *e) {
	/*
	if(!triangulate & plot) {
		clip_line(e);
	}
	if(!triangulate & !plot)
	{	printf("e %d", e->edgenbr);
		printf(" %d ", e->ep[le] != (struct Site *)NULL ? e->ep[le]->sitenbr : -1);
		printf("%d\n", e->ep[re] != (struct Site *)NULL ? e->ep[re]->sitenbr : -1);
	};
	*/
}
/////////////////////////////////////////////////
void Triangulator::clip_line(struct Edge *e) {
	struct Site *s1, *s2;
	float x1, x2, y1, y2;

	if (e->a == 1.0 && e->b >= 0.0) {
		s1 = e->ep[1];
		s2 = e->ep[0];
	}
	else {
		s1 = e->ep[0];
		s2 = e->ep[1];
	}

	if (e->a == 1.0) {
		y1 = pymin;
		if (s1 != (struct Site *)NULL && s1->coord.y > pymin) { y1 = s1->coord.y; }
		if (y1 > pymax) { return; }
		x1 = e->c - e->b * y1;
		y2 = pymax;
		if (s2 != (struct Site *)NULL && s2->coord.y < pymax) { y2 = s2->coord.y; }
		if (y2 < pymin) { return; }
		x2 = e->c - e->b * y2;

		if ((x1 > pxmax && x2 > pxmax) || (x1 < pxmin && x2 < pxmin)) { return; }
		if (x1 > pxmax) { x1 = pxmax; y1 = (e->c - x1) / e->b; }
		if (x1 < pxmin) { x1 = pxmin; y1 = (e->c - x1) / e->b; }
		if (x2 > pxmax) { x2 = pxmax; y2 = (e->c - x2) / e->b; }
		if (x2 < pxmin) { x2 = pxmin; y2 = (e->c - x2) / e->b; }
	}
	else {
		x1 = pxmin;
		if (s1 && (s1->coord.x > pxmin)) { x1 = s1->coord.x; }
		if (x1 > pxmax) { return; }
		y1 = e->c - e->a * x1;
		x2 = pxmax;
		if (s2 && (s2->coord.x < pxmax)) { x2 = s2->coord.x; }
		if (x2 < pxmin) { return; }
		y2 = e->c - e->a * x2;
		if ((y1 > pymax && y2 > pymax) || (y1 < pymin && y2 < pymin)) { return; }
		if (y1 > pymax) { y1 = pymax; x1 = (e->c - y1) / e->a; }
		if (y1 < pymin) { y1 = pymin; x1 = (e->c - y1) / e->a; }
		if (y2 > pymax) { y2 = pymax; x2 = (e->c - y2) / e->a; }
		if (y2 < pymin) { y2 = pymin; x2 = (e->c - y2) / e->a; }
	};
	line(x1, y1, x2, y2);
}
/////////////////////////////////////////////////
void Triangulator::line(int x1, int y1, int x2, int y2) {
	printf("visualization of line not implemented");
}
/////////////////////////////////////////////////
void Triangulator::ELinitialize() {
	int i;
	freeinit(&hfl, sizeof **ELhash);
	ELhashsize = 2 * sqrt_nsites;
	ELhash = (struct Halfedge **) myalloc(sizeof *ELhash * ELhashsize);
	for (i = 0; i < ELhashsize; i++) { ELhash[i] = (struct Halfedge *)NULL; }
	ELleftend = HEcreate((struct Edge *)NULL, 0);
	ELrightend = HEcreate((struct Edge *)NULL, 0);
	ELleftend->ELleft = (struct Halfedge *)NULL;
	ELleftend->ELright = ELrightend;
	ELrightend->ELleft = ELleftend;
	ELrightend->ELright = (struct Halfedge *)NULL;
	ELhash[0] = ELleftend;
	ELhash[ELhashsize - 1] = ELrightend;
}
/////////////////////////////////////////////////
void Triangulator::ELinsert(struct	Halfedge *lb, struct Halfedge *newone) {
	newone->ELleft = lb;
	newone->ELright = lb->ELright;
	lb->ELright->ELleft = newone;
	lb->ELright = newone;
}
/////////////////////////////////////////////////
struct Halfedge *Triangulator::HEcreate(struct Edge *e, int pm) {
	struct Halfedge *answer;
	answer = (struct Halfedge *) getfree(&hfl);
	answer->ELedge = e;
	answer->ELpm = pm;
	answer->PQnext = (struct Halfedge *) NULL;
	answer->vertex = (struct Site *) NULL;
	answer->ELrefcnt = 0;
	return answer;
}
/////////////////////////////////////////////////
/* This delete routine can't reclaim node, since pointers from hash
   table may be present.   */
void Triangulator::ELdelete(struct Halfedge *he) {
	he->ELleft->ELright = he->ELright;
	he->ELright->ELleft = he->ELleft;
	he->ELedge = (struct Edge *)DELETED;
}
/////////////////////////////////////////////////
struct Halfedge	* Triangulator::ELright(struct Halfedge *he) {
	return he->ELright;
}
/////////////////////////////////////////////////
struct Halfedge	* Triangulator::ELleft(struct Halfedge *he) {
	return he->ELleft;
}
/////////////////////////////////////////////////
void Triangulator::PQinsert(struct Halfedge *he, struct Site *v, float offset) {
	struct Halfedge *last, *next;

	he->vertex = v;
	ref(v);
	he->ystar = v->coord.y + offset;
	last = &PQhash[PQbucket(he)];
	while ((next = last->PQnext) != (struct Halfedge *) NULL &&
		(he->ystar > next->ystar ||
		(he->ystar == next->ystar && v->coord.x > next->vertex->coord.x))) {
		last = next;
	}
	he->PQnext = last->PQnext;
	last->PQnext = he;
	PQcount++;
}
/////////////////////////////////////////////////
void Triangulator::PQdelete(struct Halfedge *he) {
	struct Halfedge *last;
	if (he->vertex) {
		last = &PQhash[PQbucket(he)];
		while (last->PQnext != he) { last = last->PQnext; }
		last->PQnext = he->PQnext;
		PQcount--;
		deref(he->vertex);
		he->vertex = (struct Site *) NULL;
	}
}
/////////////////////////////////////////////////
int Triangulator::PQbucket(struct Halfedge *he) {
	int bucket;
	bucket = (he->ystar - ymin) / deltay * PQhashsize;
	if (bucket < 0) { bucket = 0; }
	if (bucket >= PQhashsize) { bucket = PQhashsize - 1; }
	if (bucket < PQmin) { PQmin = bucket; }
	return bucket;
}
/////////////////////////////////////////////////
int Triangulator::PQempty() {
	return PQcount == 0;
}
/////////////////////////////////////////////////
struct Point Triangulator::PQ_min() {
	struct Point answer;
	while (!PQhash[PQmin].PQnext) { PQmin++; }
	answer.x = PQhash[PQmin].PQnext->vertex->coord.x;
	answer.y = PQhash[PQmin].PQnext->ystar;
	return answer;
}
/////////////////////////////////////////////////
/* Get entry from hash table, pruning any deleted nodes */
struct Halfedge *Triangulator::ELgethash(int b) {
	struct Halfedge *he;
	if (b < 0 || b >= ELhashsize) { return (struct Halfedge *) NULL; }
	he = ELhash[b];
	if (he == (struct Halfedge *) NULL ||
		he->ELedge != (struct Edge *) DELETED) {
		return he;
	}

	/* Hash table points to deleted half edge.  Patch as necessary. */
	ELhash[b] = (struct Halfedge *) NULL;
	if (--(he->ELrefcnt) == 0) { makefree((Freenode*)he, &hfl); }
	return (struct Halfedge *) NULL;
}
/////////////////////////////////////////////////
/* returns 1 if p is to right of halfedge e */
int Triangulator::right_of(struct Halfedge *el, struct Point *p) {
	struct Edge *e;
	struct Site *topsite;
	int right_of_site, above, fast;
	float dxp, dyp, dxs, t1, t2, t3, yl;

	e = el->ELedge;
	topsite = e->reg[1];
	right_of_site = p->x > topsite->coord.x;
	if (right_of_site && el->ELpm == le) { return(1); }
	if (!right_of_site && el->ELpm == re) { return (0); }

	if (e->a == 1.0) {
		dyp = p->y - topsite->coord.y;
		dxp = p->x - topsite->coord.x;
		fast = 0;
		if ((!right_of_site & e->b < 0.0) | (right_of_site & e->b >= 0.0)) {
			above = dyp >= e->b*dxp;
			fast = above;
		}
		else {
			above = p->x + p->y*e->b > e->c;
			if (e->b < 0.0) { above = !above; }
			if (!above) { fast = 1; }
		}
		if (!fast) {
			dxs = topsite->coord.x - (e->reg[0])->coord.x;
			above = e->b * (dxp*dxp - dyp * dyp) <
				dxs*dyp*(1.0 + 2.0*dxp / dxs + e->b*e->b);
			if (e->b < 0.0) { above = !above; }
		}
	}
	else { // e->b==1.0
		yl = e->c - e->a*p->x;
		t1 = p->y - yl;
		t2 = p->x - topsite->coord.x;
		t3 = yl - topsite->coord.y;
		above = t1 * t1 > t2*t2 + t3 * t3;
	}
	return el->ELpm == le ? above : !above;
}
/////////////////////////////////////////////////
struct Halfedge *Triangulator::ELleftbnd(struct Point *p) {
	int i, bucket;
	struct Halfedge *he;

	/* Use hash table to get close to desired halfedge */
	bucket = (p->x - xmin) / deltax * ELhashsize;
	if (bucket < 0) { bucket = 0; }
	if (bucket >= ELhashsize) { bucket = ELhashsize - 1; }
	he = ELgethash(bucket);
	if (he == (struct Halfedge *) NULL) {
		for (i = 1; 1; i++) {
			if ((he = ELgethash(bucket - i)) != (struct Halfedge *) NULL) { break; }
			if ((he = ELgethash(bucket + i)) != (struct Halfedge *) NULL) { break; }
		}
		totalsearch += i;
	}
	ntry++;
	/* Now search linear list of halfedges for the corect one */
	if (he == ELleftend || (he != ELrightend && right_of(he, p))) {
		do { he = he->ELright; } while (he != ELrightend && right_of(he, p));
		he = he->ELleft;
	}
	else {
		do { he = he->ELleft; } while (he != ELleftend && !right_of(he, p));
	}

	/* Update hash table and reference counts */
	if (bucket > 0 && bucket < ELhashsize - 1) {
		if (ELhash[bucket]) { --(ELhash[bucket]->ELrefcnt); }
		ELhash[bucket] = he;
		++(ELhash[bucket]->ELrefcnt);
	};
	return he;
}

/////////////////////////////////////////////////
struct Site *Triangulator::leftreg(struct Halfedge *he) {
	if (he->ELedge == (struct Edge *)NULL) { return(bottomsite); }
	return he->ELpm == le ? he->ELedge->reg[le] : he->ELedge->reg[re];
}
/////////////////////////////////////////////////
struct Site *Triangulator::rightreg(struct Halfedge *he) {
	if (he->ELedge == (struct Edge *)NULL) { return(bottomsite); }
	return he->ELpm == le ? he->ELedge->reg[re] : he->ELedge->reg[le];
}
/////////////////////////////////////////////////
void Triangulator::deref(struct Site *v) {
	--(v->refcnt);
	if (v->refcnt == 0) { makefree((Freenode*)v, &sfl); }
}
/////////////////////////////////////////////////
void Triangulator::ref(struct	Site *v) {
	++(v->refcnt);
}
/////////////////////////////////////////////////
struct Edge *Triangulator::bisect(struct Site *s1, struct Site*s2) {
	float dx, dy, adx, ady;
	struct Edge *newedge;
	newedge = (struct Edge *) getfree(&efl);

	newedge->reg[0] = s1;
	newedge->reg[1] = s2;
	ref(s1);
	ref(s2);
	newedge->ep[0] = (struct Site *) NULL;
	newedge->ep[1] = (struct Site *) NULL;

	dx = s2->coord.x - s1->coord.x;
	dy = s2->coord.y - s1->coord.y;
	adx = dx > 0 ? dx : -dx;
	ady = dy > 0 ? dy : -dy;
	newedge->c = s1->coord.x * dx + s1->coord.y * dy + (dx*dx + dy * dy)*0.5;
	if (adx > ady) {
		newedge->a = 1.0; newedge->b = dy / dx; newedge->c /= dx;
	}
	else {
		newedge->b = 1.0; newedge->a = dx / dy; newedge->c /= dy;
	}

	newedge->edgenbr = nedges;
	out_bisector(newedge);
	nedges++;
	return newedge;
}
/////////////////////////////////////////////////
struct Site *Triangulator::intersect(struct Halfedge *el1,
	struct Halfedge *el2, struct Point *p) {
	struct Edge *e1, *e2, *e;
	struct Halfedge *el;
	float d, xint, yint;
	int right_of_site;
	struct Site *v;

	e1 = el1->ELedge;
	e2 = el2->ELedge;
	if (!e1 || !e2) { return NULL; }
	if (e1->reg[1] == e2->reg[1]) { return NULL; }

	d = e1->a * e2->b - e1->b * e2->a;
	if (-1.0e-10 < d && d < 1.0e-10) { return NULL; }

	xint = (e1->c*e2->b - e2->c*e1->b) / d;
	yint = (e2->c*e1->a - e1->c*e2->a) / d;

	if ((e1->reg[1]->coord.y < e2->reg[1]->coord.y) ||
		(e1->reg[1]->coord.y == e2->reg[1]->coord.y &&
			e1->reg[1]->coord.x < e2->reg[1]->coord.x)) {
		el = el1;
		e = e1;
	}
	else {
		el = el2;
		e = e2;
	}
	right_of_site = xint >= e->reg[1]->coord.x;
	if ((right_of_site && el->ELpm == le) ||
		(!right_of_site && el->ELpm == re)) {
		return NULL;
	}

	v = (struct Site *) getfree(&sfl);
	v->refcnt = 0;
	v->coord.x = xint;
	v->coord.y = yint;
	return v;
}
/////////////////////////////////////////////////
float Triangulator::dist(struct Site *s, struct Site *t) {
	float dx, dy;
	dx = s->coord.x - t->coord.x;
	dy = s->coord.y - t->coord.y;
	return sqrt(dx*dx + dy * dy);
}
/////////////////////////////////////////////////
void Triangulator::makevertex(struct Site *v) {
	v->sitenbr = nvertices;
	++nvertices;
	out_vertex(v);
}
/////////////////////////////////////////////////
void Triangulator::endpoint(struct Edge *e, int lr, struct Site *s) {
	e->ep[lr] = s;
	ref(s);
	if (!(e->ep[re - lr])) { return; }
	out_ep(e);
	deref(e->reg[le]);
	deref(e->reg[re]);
	makefree((Freenode*)e, &efl);
}
/////////////////////////////////////////////////

void Triangulator::voronoi() {
	struct Site *newsite, *bot, *top, *temp, *p;
	struct Site *v;
	struct Point newintstar;
	int pm;
	struct Halfedge *lbnd, *rbnd, *llbnd, *rrbnd, *bisector;
	struct Edge *e;

	PQinitialize();
	bottomsite = next();
	out_site(bottomsite);
	ELinitialize();

	newsite = next();
	while (1) {
		if (!PQempty()) newintstar = PQ_min();
		if (newsite &&
			(PQempty() ||
				newsite->coord.y < newintstar.y ||
				(newsite->coord.y == newintstar.y &&
					newsite->coord.x < newintstar.x))) { // new site is smallest

			out_site(newsite);
			lbnd = ELleftbnd(&(newsite->coord));
			rbnd = ELright(lbnd);
			bot = rightreg(lbnd);
			e = bisect(bot, newsite);
			bisector = HEcreate(e, le);
			ELinsert(lbnd, bisector);
			if ((p = intersect(lbnd, bisector)) != NULL) {
				PQdelete(lbnd);
				PQinsert(lbnd, p, dist(p, newsite));
			}
			lbnd = bisector;
			bisector = HEcreate(e, re);
			ELinsert(lbnd, bisector);
			if ((p = intersect(bisector, rbnd)) != NULL) {
				PQinsert(bisector, p, dist(p, newsite));
			};

			newsite = next();
		}
		else if (!PQempty()) {  // intersection is smallest
			lbnd = PQextractmin();
			llbnd = ELleft(lbnd);
			rbnd = ELright(lbnd);
			rrbnd = ELright(rbnd);
			bot = leftreg(lbnd);
			top = rightreg(rbnd);

			out_triple(bot, top, rightreg(lbnd));
			v = lbnd->vertex;
			makevertex(v);
			endpoint(lbnd->ELedge, lbnd->ELpm, v);
			endpoint(rbnd->ELedge, rbnd->ELpm, v);
			ELdelete(lbnd);
			PQdelete(rbnd);
			ELdelete(rbnd);
			pm = le;
			if (bot->coord.y > top->coord.y) {
				temp = bot;
				bot = top;
				top = temp;
				pm = re;
			}
			e = bisect(bot, top);
			bisector = HEcreate(e, pm);
			ELinsert(llbnd, bisector);
			endpoint(e, re - pm, v);
			deref(v);
			if ((p = intersect(llbnd, bisector)) != NULL) {
				PQdelete(llbnd);
				PQinsert(llbnd, p, dist(p, bot));
			}
			if ((p = intersect(bisector, rrbnd)) != NULL) {
				PQinsert(bisector, p, dist(p, bot));
			};
		}
		else {
			break;
		}
	}

	for (lbnd = ELright(ELleftend); lbnd != ELrightend; lbnd = ELright(lbnd)) {
		e = lbnd->ELedge;
		out_ep(e);
	}
}

/*END DELAUNEY*/



/*****************************************************************************/
/* BEGIN SOURCES FOR GEOMETRIIC VERIFICATION...*/
/*****************************************************************************/
/*****************************************************************************/
class Transformation2D {
public:
	Transformation2D() { nDefiningCorrespondences = 0; }
	Transformation2D(const Transformation2D& s) {
		nDefiningCorrespondences = s.nDefiningCorrespondences;
		bDefined = s.bDefined;
		bInversionValid = s.bInversionValid;
	}
	Transformation2D& operator=(const Transformation2D&s) {
		nDefiningCorrespondences = s.nDefiningCorrespondences;
		bDefined = s.bDefined;
		bInversionValid = s.bInversionValid;
		return *this;
	}
	~Transformation2D() { }
	virtual void DefineAsIdentity(void) = 0;
	virtual bool defineFromPointCorrespondences(DV2* a,
		DV2* b,
		bool bCalcInverse = true) = 0;
	virtual void setTo(const Transformation2D* pT) {
		// Makes this transformation being identical to *pT in any respect.
		bDefined = pT->bDefined;
		bInversionValid = pT->bInversionValid;
	}
	virtual DV2 out(DV2 p) = 0;
	virtual DV2 in(DV2 p) = 0;
	virtual DV2 vout(DV2 v) = 0;
	virtual DV2 vin(DV2 v) = 0;
	virtual int getVariables(double*& x) = 0;
	virtual int getHStepPattern(double*& h) = 0;
	// For below, caller is responsible for cleanup, NULL if impossible
	virtual Transformation2D*createInverse() = 0;
protected:
	int nDefiningCorrespondences;
	bool bDefined;
	bool bInversionValid;
};
/*****************************************************************************/
class ATrans2 : public Transformation2D, public DAT2 {
public:
	ATrans2(void) {
		bDefined = false;
		nDefiningCorrespondences = 3;
		h[0] = 0.1;
		h[1] = 0.1;
		h[2] = 0.1;
		h[3] = 0.1;
		h[4] = 0.1;
		h[5] = 0.1;
	}
	ATrans2(const ATrans2& s) : Transformation2D(s), DAT2(s) {
		h[0] = s.h[0];
		h[1] = s.h[1];
		h[2] = s.h[2];
		h[3] = s.h[3];
		h[4] = s.h[4];
		h[5] = s.h[5];
	}
	ATrans2& operator=(const ATrans2& s) {
		(*(Transformation2D*)this) = s;
		a[0] = s.a[0];
		a[1] = s.a[1];
		a[2] = s.a[2];
		a[3] = s.a[3];
		t = s.t;
		h[0] = s.h[0];
		h[1] = s.h[1];
		h[2] = s.h[2];
		h[3] = s.h[3];
		h[4] = s.h[4];
		h[5] = s.h[5];
		return *this;
	}
	virtual void setTo(const Transformation2D* pT) {
		Transformation2D::setTo(pT);
		ATrans2&at = *(ATrans2*)pT;
		a[0] = at.a[0];
		a[1] = at.a[1];
		a[2] = at.a[2];
		a[3] = at.a[3];
		t.x = at.t.x;
		t.y = at.t.y;
	}
	bool operator==(const ATrans2& other) const {
		return DAT2::operator==(other);
	}
	bool DefineFromTriangles(DV2 triFrom[3], DV2 triTo[3]) {
		bDefined = defineFromTriangles(triFrom, triTo);
		return bDefined;
	}
	bool DefineFromTrianglesV2(V2 triFrom[3], V2 triTo[3]) {
		bDefined = false;
		float ax = triFrom[0].x;
		float ay = triFrom[0].y;
		float bx = triFrom[1].x;
		float by = triFrom[1].y;
		float cx = triFrom[2].x;
		float cy = triFrom[2].y;
		float A = ax * (by - cy) + bx * (cy - ay) + cx * (ay - by);//TODO:optimize see a**
		if (A == 0) { return false; };
		float Ai = 1.0f / A;

		float axs = triTo[0].x;
		float ays = triTo[0].y;
		float bxs = triTo[1].x;
		float bys = triTo[1].y;
		float cxs = triTo[2].x;
		float cys = triTo[2].y;

		DV3 t_bary(bx*cy - cx * by, cx*ay - ax * cy, ax*by - ay * bx);

		DV3 Q1(by - cy, cy - ay, ay - by);//a**
		DV3 Q2(cx - bx, ax - cx, bx - ax);
		DV3 v1(axs, bxs, cxs);
		DV3 v2(ays, bys, cys);
		v1 *= Ai;
		v2 *= Ai;

		a[0] = v1 * Q1;
		a[1] = v1 * Q2;
		a[2] = v2 * Q1;
		a[3] = v2 * Q2;

		t.x = v1 * t_bary;
		t.y = v2 * t_bary;
		bDefined = true;
		return true;
	}

	virtual bool defineFromPointCorrespondences(DV2* a,
		DV2* b,
		bool bCalcInverse = true) {
		return DefineFromTriangles(a, b);
	}
	virtual int getVariables(double*& x) {
		x = a;
		return 6;
	}
	virtual void DefineAsIdentity(void) {
		a[0] = 1.0f;
		a[1] = 0.0f;
		a[2] = 0.0f;
		a[3] = 1.0f;
		t.x = 0;
		t.y = 0;
		bDefined = true;
	}
	virtual DV2 out(DV2 p) {
		return DAT2::out(p);
	}
	DV2 vout(DV2 v) {
		return DAT2::vout(v);
	}
	virtual DV2 in(DV2 p) {
		printf("not implemented");
		return DV2();
	}
	DV2 vin(DV2 v) {
		printf("not implemented");
		return DV2();
	}
	bool CalcInverseTransformation(ATrans2& at) {
		at.bDefined = calcInverseTransformation(at);
		return at.bDefined;
	}
	double phiOut(double phi) {
		return DAT2::phiOut(phi);
	}
	// DV2phi v2phiOut(DV2phi p);
	float CalcError(ATrans2& c) {
		return DAT2::calcError(c);
	}
	void DefineAsRigidTransformation(float phi, DV2 nt) {
		defineAsRigidTransformation(phi, nt);
		bDefined = true;
	}
	void DefineAsScaledRigidTransformation(float scale, float phi, DV2 nt) {
		defineAsScaledRigidTransformation(scale, phi, nt);
		bDefined = true;
	}
	bool DefineAsScaledRigidTransformation(DV2 fromA, DV2 fromB,
		DV2 toA, DV2 toB) {
		bDefined = defineAsScaledRigidTransformation(fromA, fromB, toA, toB);
		return bDefined;
	}
	/*
	void DefineAsRigidTransformationAround(DV2 center, float dphi, DV2 dt) {
		defineAsRigidTransformationAround(center,dphi,dt);
		bDefined = true;
	}
	*/
	float getAngularDeviationFromBeingOrthogonal() {
		return DAT2::getAngularDeviationFromBeingOrthogonal();
	}
	float GetRigidity(void) {
		if (!bDefined) { return 0.0f; }
		return getRigidity();
	}
	bool getLengthOfAxes(double& lu, double& lv) {
		if (!bDefined) { return false; }
		getLengthOfAxes(lu, lv);
		return true;
	}
	virtual int getHStepPattern(double*& h_) {
		h_ = h;
		return 6;
	}
	virtual Transformation2D*createInverse() {
		if (!bDefined) {
			printf("affine transformation that is to be inverted is undefined");
			return NULL;
		}
		ATrans2*pAI = new ATrans2();
		if (!CalcInverseTransformation(*pAI)) {
			printf("failed to determine inverse transformation");
			return NULL;
		}
		return pAI;
	}
	double h[6];  // Do not add member variables before h
};
/*****************************************************************************/

class Homography : public Transformation2D {
public:
	Homography(const Homography& s) :
		tinyDet(s.tinyDet),
		thresh_allowedDefinitionDist(s.thresh_allowedDefinitionDist),
		bAvoidPlaneFlippers(s.bAvoidPlaneFlippers) {
		ini();
		set(s.x[0], s.x[1], s.x[2], s.x[3], s.x[4], s.x[5], s.x[6], s.x[7], s.x[8],
			s.bInversionValid);
	}
	Homography(double tinyDet = 1e-5, double thresh_allowedDefinitionDist = 0.01,
		bool bAvoidPlaneFlippers = true) :
		tinyDet(tinyDet),
		thresh_allowedDefinitionDist(thresh_allowedDefinitionDist),
		bAvoidPlaneFlippers(bAvoidPlaneFlippers) {
		ini();
	}
	Homography(double h00, double h01, double h02,
		double h10, double h11, double h12,
		double h20, double h21, double h22,
		bool bCalcInversion = true, double tinyDet = 1e-5,
		double thresh_allowedDefinitionDist = 0.01,
		bool bAvoidPlaneFlippers = true) :
		tinyDet(tinyDet),
		thresh_allowedDefinitionDist(thresh_allowedDefinitionDist),
		bAvoidPlaneFlippers(bAvoidPlaneFlippers) {
		ini();
		set(h00, h01, h02, h10, h11, h12, h20, h21, h22, bCalcInversion);
	}
	~Homography() {
		delete pA;
		delete pAi;
		delete pX;
		delete pB;
		delete pH;
		delete pHi;
		delete p3;
		delete r3;
	}
	virtual bool defineFromPointCorrespondences(DV2*morg,
		DV2*morg_,
		bool bCalcInversion = true) {
		if (bAvoidPlaneFlippers&&flipsPlane(morg, morg_)) { return false; }
		// See homography.pdf for a documenation of the names of the variables
		// and a mathematical derivation of the underlying equations
		// to improve numerical precision define local coordinate systems in
		// the source and target space
		localOrigin.x = 0;
		localOrigin.y = 0;
		localOrigin_.x = 0;
		localOrigin_.y = 0;

		//localOrigin=0.25*(morg[0]+morg[1]+morg[2]+morg[3]);
		//localOrigin_=0.25*(morg_[0]+morg_[1]+morg_[2]+morg_[3]);
		DV2 m[4];
		DV2 m_[4];
		for (int i = 0; i < 4; i++) {
			m[i] = morg[i] - localOrigin;
			m_[i] = morg_[i] - localOrigin_;
		}
		bInversionValid = false;
		bDefined = false;
#define u0 m[0].x
#define v0 m[0].y
#define u1 m[1].x
#define v1 m[1].y
#define u2 m[2].x
#define v2 m[2].y
#define u3 m[3].x
#define v3 m[3].y
#define u0_ m_[0].x
#define v0_ m_[0].y
#define u1_ m_[1].x
#define v1_ m_[1].y
#define u2_ m_[2].x
#define v2_ m_[2].y
#define u3_ m_[3].x
#define v3_ m_[3].y

		double* aa = a;
		*aa++ = 0; *aa++ = 0; *aa++ = 0; *aa++ = -u0; *aa++ = -v0; *aa++ = -1.0; *aa++ = u0 * v0_; *aa++ = v0 * v0_;
		*aa++ = u0; *aa++ = v0; *aa++ = 1.0; *aa++ = 0; *aa++ = 0; *aa++ = 0; *aa++ = -u0 * u0_; *aa++ = -v0 * u0_;
		*aa++ = 0; *aa++ = 0; *aa++ = 0; *aa++ = -u1; *aa++ = -v1; *aa++ = -1.0; *aa++ = u1 * v1_; *aa++ = v1 * v1_;
		*aa++ = u1; *aa++ = v1; *aa++ = 1.0; *aa++ = 0; *aa++ = 0; *aa++ = 0; *aa++ = -u1 * u1_; *aa++ = -v1 * u1_;
		*aa++ = 0; *aa++ = 0; *aa++ = 0; *aa++ = -u2; *aa++ = -v2; *aa++ = -1.0; *aa++ = u2 * v2_; *aa++ = v2 * v2_;
		*aa++ = u2; *aa++ = v2; *aa++ = 1.0; *aa++ = 0; *aa++ = 0; *aa++ = 0; *aa++ = -u2 * u2_; *aa++ = -v2 * u2_;
		*aa++ = 0; *aa++ = 0; *aa++ = 0; *aa++ = -u3; *aa++ = -v3; *aa++ = -1.0; *aa++ = u3 * v3_; *aa++ = v3 * v3_;
		*aa++ = u3; *aa++ = v3; *aa++ = 1.0; *aa++ = 0; *aa++ = 0; *aa++ = 0; *aa++ = -u3 * u3_; *aa = -v3 * u3_;

		b[0] = -v0_; b[1] = u0_; b[2] = -v1_; b[3] = u1_; b[4] = -v2_; b[5] = u2_; b[6] = -v3_; b[7] = u3_;

		double det = cvInvert(pA, pAi, CV_LU);
		if (fabs(det) <= tinyDet) {
			return false;
		}
		cvMatMul(pAi, pB, pX);
		bDefined = true;  // Set flag to allow mapping temporarily
		// Verify whether the original points are mapped correctly
		double worstDist = 0;
		for (int i = 0; i < 4; i++) {
			DV2 po = out(morg[i]);
			double curDist = (po - morg_[i]).len();
			if (curDist > worstDist) {
				worstDist = curDist;
			}
		}
		if (worstDist > thresh_allowedDefinitionDist) {
#ifndef NDEBUG
			printf("homography defined too imprecisely\n");
#endif
			bDefined = false;
			return false;
		}

		/*#ifndef NDEBUG
		   double mem_check_is_b[8];
		   CvMat cvCheck_is_b=cvMat(8,1,CV_64FC1,mem_check_is_b);
		   cvMatMul(pA,pX,&cvCheck_is_b);
		#endif*/
		if (bCalcInversion) {
			tryCalcInversion();
		}
		bDefined = true;
		return true;
	}

	virtual void DefineAsIdentity(void) {
		memset(x, 0, 8 * sizeof(*x));
		x[0] = 1.0;
		x[4] = 1.0;
		//x[8] is already 1.0
		bDefined = true;
		memset(hi, 0, 9 * sizeof(*hi));
		hi[0] = 1.0;
		hi[4] = 1.0;
		hi[8] = 1.0;
		bInversionValid = true;
	}
	void ini() {
		nDefiningCorrespondences = 4;
		bDefined = false;
		bInversionValid = false;

		// The new constructs here only allocate the headers not the data.
		// Allocation is done in this way to avoid the need to include
		// respective heades in the header of this file.
		pA = new CvMat();
		pAi = new CvMat();
		pX = new CvMat();
		pB = new CvMat();
		pH = new CvMat();
		pHi = new CvMat();
		p3 = new CvMat();
		r3 = new CvMat();

		// The initialization of the matrices is somehow unconventional, but is done so on purpsoe.
		// The tricky part of the code is that the memory block double x[9] (see the header file)
		// is used as underlying memory for two different matrices.
		// First only the first 8 of the 9 elemenets are used for the 8x1 vector pX which receives the result of a matrix multiplication as a solution of a system of linear equations.
		// Second the same data with an aditional element initialized to 1.0 is used as a 3x3 matrix to acutally project points.
		*pA = cvMat(8, 8, CV_64FC1, a);
		*pAi = cvMat(8, 8, CV_64FC1, ai);
		*pX = cvMat(8, 1, CV_64FC1, x);
		*pB = cvMat(8, 1, CV_64FC1, b);
		*pH = cvMat(3, 3, CV_64FC1, x);//duplicate use of x intentionally!
		*pHi = cvMat(3, 3, CV_64FC1, hi);
		*p3 = cvMat(3, 1, CV_64FC1, p);
		*r3 = cvMat(3, 1, CV_64FC1, r);
		x[8] = 1.0;
		p[2] = 1.0;
		r[2] = 1.0;

		hstep[0] = 0.01;
		hstep[1] = 0.01;
		hstep[2] = 0.01;
		hstep[3] = 0.01;
		hstep[4] = 0.01;
		hstep[5] = 0.01;
		hstep[6] = 0.000001;
		// @TODO: I believe that there is an error in how the hstp pattern is set,
		// because I think that the 9 components x of the homography when seen as
		// a matrix, are stored rowwise and the translation vector would then be
		// at indices [2] and [5]
		hstep[7] = 0.000001;
	}
	void set(double h00, double h01, double h02,
		double h10, double h11, double h12,
		double h20, double h21, double h22,
		bool bCalcInversion = true) {
		x[0] = h00;
		x[1] = h01;
		x[2] = h02;
		x[3] = h10;
		x[4] = h11;
		x[5] = h12;
		x[6] = h20;
		x[7] = h21;
		x[8] = h22;
		if (bCalcInversion) {
			tryCalcInversion();
		}
		bDefined = true;
	}
	void tryCalcInversion() {
		double detHi = cvInvert(pH, pHi, CV_LU);
		if (fabs(detHi) > tinyDet) {
			bInversionValid = true;
		}
	}
	bool flipsPlane(DV2* a, DV2* b) {
		for (int i = 0; i < 4; i++) {
			DV2 pa = a[i];
			DV2 pb = b[i];
			for (int k = i + 1; k < 4; k++) {
				DV2 va = a[k] - pa;
				DV2 vb = b[k] - pb;
				DV2 na = ~va;
				DV2 nb = ~vb;
				for (int j = 0; j < 4; j++) {
					if (j != i && j != k) {
						DV2 ta = a[j] - pa;
						DV2 tb = b[j] - pb;
						double da = ta * na;
						double db = tb * nb;
						if (da*db < 0) {
							return true;
						}
					}
				}
			}
		}
		return false;
	}
	virtual int getVariables(double*& x_) {
		x_ = x;
		return 8;
	}
	virtual DV2 out(DV2 m) {
		assert(bDefined);
		p[0] = m.x - localOrigin.x;
		p[1] = m.y - localOrigin.y;  //last homogeneous coordinate p[2]==1.0 (see constructor)
		cvMatMul(pH, p3, r3);
		double w = r[2];
		return localOrigin_ + DV2(r[0] / w, r[1] / w);
	}
	virtual DV2 in(DV2 m_) {
		assert(bDefined);
		assert(bInversionValid);
		p[0] = m_.x - localOrigin_.x;
		p[1] = m_.y - localOrigin_.y;  //last homogeneous coordinate p[2]==1.0 (see constructor)
		cvMatMul(pHi, p3, r3);
		double w = r[2];
		return localOrigin + DV2(r[0] / w, r[1] / w);
	}
	virtual int getHStepPattern(double*& h_) {
		h_ = hstep;
		return 8;
	}
	virtual void setTo(const Transformation2D* pT) {
		// Makes this transformation being identical to *pT in any respect.
		Transformation2D::setTo(pT);
		Homography&ht = *(Homography*)pT;
		tinyDet = ht.tinyDet;
		thresh_allowedDefinitionDist = ht.thresh_allowedDefinitionDist;
		bAvoidPlaneFlippers = ht.bAvoidPlaneFlippers;
		localOrigin = ht.localOrigin;
		localOrigin_ = ht.localOrigin_;
		memcpy(a, ht.a, sizeof(a));
		memcpy(ai, ht.ai, sizeof(ai));
		memcpy(b, ht.b, sizeof(b));
		memcpy(x, ht.x, sizeof(x));
		memcpy(hi, ht.hi, sizeof(hi));
		memcpy(hstep, ht.hstep, sizeof(hstep));
	}
	virtual Transformation2D*createInverse() {
		// Caller is responsible for cleanup, NULL if impossible
		if (!bDefined) {
			printf("homography that is to be inverted is undefined\n");
			return NULL;
		}
		if (!bInversionValid) { tryCalcInversion(); }
		if (!bInversionValid) {
			printf("failed to invert homography\n");
			return NULL;
		}
		Homography*pHI = new Homography(tinyDet, thresh_allowedDefinitionDist,
			bAvoidPlaneFlippers);
		bool bCalcInversion = true;  //(inversion of inversion)
		pHI->set(hi[0], hi[1], hi[2], hi[3], hi[4], hi[5], hi[6], hi[7], hi[8], true);
		return pHI;
	}
	virtual DV2 vout(DV2 v) { printf("not implemented\n"); return DV2(); }
	virtual DV2 vin(DV2 v) { printf("not implemented\n"); return DV2(); }
	DV2 localOrigin;
	DV2 localOrigin_;
	double a[8 * 8];
	double ai[8 * 8];
	double b[8];
	double x[9];  // Intentionally 1 larger! 9 components of homography-matrix
	double hi[9];
	double p[3];
	double r[3];
	double hstep[8];
	class CvMat*pA;
	class CvMat*pAi;
	class CvMat*pX;
	class CvMat*pB;
	class CvMat*pH;
	class CvMat*pHi;
	class CvMat*p3;
	class CvMat*r3;
	double tinyDet;
	double thresh_allowedDefinitionDist;
	bool bAvoidPlaneFlippers;
};

/*****************************************************************************/
class RansacKernel {
public:
	RansacKernel() {
		pCachedParts = NULL;
		nCachedParts = 0;
		nPartsForOneModel = 0;
		nBytesForOnePart = 0;
		nBytesForOneModel = 0;
	}
	virtual ~RansacKernel() {
		releasePartsMemory();
	}
	virtual bool cacheParts(std::vector<cv::KeyPoint> kp[2],
		std::vector< cv::DMatch >& candidate_matches,
		bool bUseGlobalFrame = false,
		bool bUseUndistortedData = false) {
		return false;
	}
	/**
	* @fn construct constructs a model from n parts. (n=nPartsForOneModel())
	* The parts are provides in terms of an array of indices into the pParts array.
	* The constructed model is stored to *pModel
	* return true, if construction was sucessfull.
	*/
	virtual bool construct(void* pModel, int* partIndices) = 0;
	/**
	 *called fore each model before memory for all models is released.
	 *Does not need to be implemented in many cases, but is usefull
	 *for destructing (in place constructed objects).
	*/
	virtual void destruct(void* pModel) {};
	/**
	* return whether part iPart in the cached data is in consensus with model pModel
	*/
	virtual bool accept(void* pModel, int iPart) = 0;
	void releasePartsMemory() {
		if (pCachedParts) {
			free(pCachedParts);
			pCachedParts = NULL;
		}
	}
	/**
	*gets a pointe to the parts (basically the pParts pointer)
	*/
	void* cachedParts() { return pCachedParts; }
	/**
	* gets the number of parts cachedn
	*/
	int getNumCachedParts() { return nCachedParts; }
	/**
	* return the number of parts required to construct one model.
	* returns the member variable nPartsForModel basically.
	*/
	int getNumPartsForOneModel() { return nPartsForOneModel; }
	/*
	* number of bytes required to store one part.
	* basically returns the member variable nBytesForOneModel
	*/
	int partSize() { return nBytesForOnePart; }
	/*
	* number of bytes required to store one model;
	* basically return the member variable nBytesForOneModel
	*/
	int modelSize() { return nBytesForOneModel; }
	/*
	* returns succces
	* pTransformation2D: a 2D transformation representing the model
	* the caller is responsible for the returned objekt
	* pModel: the model
	*/
	virtual bool getTransformation2D(class Transformation2D*& pTransformation2D,
		void*pModel) {
		return false;
	}
protected:
	void* pCachedParts;  // Pointer to cached parts
			 // NULL if no parts cached or memory released.
	int nCachedParts;       // Number of parts cached one after another in
				// pCachedParts, each having size partSize()
	int nPartsForOneModel;  // Number of parts required to construct one model.
	int nBytesForOnePart;
	int nBytesForOneModel;
};
/*****************************************************************************/
class RansacKernel_2DPointTransformation : public RansacKernel {
public:
	RansacKernel_2DPointTransformation(float thresh_dist = 4.0f) :
		thresh_dist(thresh_dist) {
		nBytesForOnePart = sizeof(TwoV2);
		nBytesForOneModel = 0;
		nPartsForOneModel = 0;  // Overwrite the last two in derived class
	}
	virtual bool cacheParts(std::vector<cv::KeyPoint> kp[2],
		std::vector< cv::DMatch >& candidate_matches,
		bool bUseGlobalFrame = false,
		bool bUseUndistortedData = false) {
		int nR = candidate_matches.size();
		releasePartsMemory();
		pCachedParts = malloc(nBytesForOnePart * nR);
		pCached2V2 = (TwoV2*)pCachedParts;
		V2 pointA, pointB;
		for (int i = 0; i < nR; i++) {
			int ia = candidate_matches[i].queryIdx;
			int ib = candidate_matches[i].trainIdx;
			V2&pointA = pCached2V2[i].a;
			V2&pointB = pCached2V2[i].b;
			pointA.x = kp[0].at(ia).pt.x;
			pointA.y = kp[0].at(ia).pt.y;
			pointB.x = kp[1].at(ib).pt.x;
			pointB.y = kp[1].at(ib).pt.y;
		}
		nCachedParts = nR;
		return true;
	}
	// virtual bool construct(void* pModel, int* partIndices) { return true; }
	// virtual bool accept(void* pModel, int iPart) { return true; }
	TwoV2* pCached2V2;  //==pCachedParts
	float thresh_dist;
};
/*****************************************************************************/
class RansacKernel_AT2 : public RansacKernel_2DPointTransformation {
public:
	RansacKernel_AT2(float thresh_dist = 4.0f) :
		RansacKernel_2DPointTransformation(thresh_dist) {
		nPartsForOneModel = 3;
		nBytesForOneModel = sizeof(AT2);
	}
	virtual bool construct(void* pModel, int* partIndices) {
		AT2* m = (AT2*)pModel;
		TwoV2& pair0 = pCached2V2[partIndices[0]];
		TwoV2& pair1 = pCached2V2[partIndices[1]];
		TwoV2& pair2 = pCached2V2[partIndices[2]];
		return m->defineFromTriangles(pair0.a, pair1.a, pair2.a,
			pair0.b, pair1.b, pair2.b);
	}
	virtual bool accept(void* pModel, int iPart) {
		AT2* m = (AT2*)pModel;
		TwoV2& pair = pCached2V2[iPart];
		V2 dif = pair.b - m->out(pair.a);
		float error = dif.len();
		return (error < thresh_dist);
	}
	virtual bool getTransformation2D(class Transformation2D*& pTransformation2D, void*pModel) {
		const AT2* pAT2 = (AT2*)pModel;
		ATrans2* pATrans2 = new ATrans2();
		pATrans2->a[0] = pAT2->a[0];
		pATrans2->a[1] = pAT2->a[1];
		pATrans2->a[2] = pAT2->a[2];
		pATrans2->a[3] = pAT2->a[3];
		pATrans2->t = pAT2->t;
		pTransformation2D = pATrans2;
		return true;
	}
};

/*****************************************************************************/
class RansacKernel_Homography : public RansacKernel_2DPointTransformation {
public:
	RansacKernel_Homography(float thresh_dist = 4.0f) :
		RansacKernel_2DPointTransformation(thresh_dist) {
		nPartsForOneModel = 4;
		nBytesForOneModel = sizeof(Homography);
	}
	virtual bool construct(void* pModel, int* partIndices) {
		// Placement constructor (not well known),
		// constructes an object at a given memory address
		Homography*h = new (pModel) Homography();
		TwoV2&pair0 = pCached2V2[partIndices[0]];
		TwoV2&pair1 = pCached2V2[partIndices[1]];
		TwoV2&pair2 = pCached2V2[partIndices[2]];
		TwoV2&pair3 = pCached2V2[partIndices[3]];
		DV2 a[4];
		a[0] = pair0.a;
		a[1] = pair1.a;
		a[2] = pair2.a;
		a[3] = pair3.a;
		DV2 b[4];
		b[0] = pair0.b;
		b[1] = pair1.b;
		b[2] = pair2.b;
		b[3] = pair3.b;
		return h->defineFromPointCorrespondences(a, b, false);
	}
	virtual void destruct(void* pModel) {
		((Homography*)pModel)->Homography::~Homography();
		//this is because we did the inplace construction in (construct).
	}
	virtual bool accept(void* pModel, int iPart) {
		Homography*m = (Homography*)pModel;
		TwoV2&pair = pCached2V2[iPart];
		V2 dif = pair.b - m->out(pair.a);
		float error = dif.len();
		return (error < thresh_dist);
	}
	virtual bool getTransformation2D(class Transformation2D*& pTransformation2D,
		void*pModel) {
		if (!pModel) { return false; }
		pTransformation2D = new Homography(*(Homography*)pModel);
		return true;
	}
};

///////////////////////////////////////////////////////////////////////////
bool consensus_better(int& a, int& b) {
	return (a > b);
}
/*****************************************************************************/
bool geometrically_verify_matches(Transformation2D*& pTransformation2D,
	std::vector< cv::DMatch >& final_matches,
	std::vector< cv::KeyPoint > kp[2],
	std::vector< cv::DMatch >& grid_matches,
	RansacKernel* pKernel,
	int nChunkSize, bool bChunkedCreation) {
	if (!pKernel) { return false; }
	if (!pKernel->cacheParts(kp, grid_matches, false, false)) { return false; }
	int nPartsForOneModel = pKernel->getNumPartsForOneModel();
	int nParts = pKernel->getNumCachedParts();
	if (!nParts) {
		printf("no parts?\n");
		return false;
	}
	int nEffectiveChunkSize;

	if (!bChunkedCreation) {
		nEffectiveChunkSize = nParts;
	}
	else {
		nEffectiveChunkSize = nChunkSize;
	}

	int nCompleteChunks = nParts / nEffectiveChunkSize;
	int nPartsOfLastIncompleteChunk = nParts % nEffectiveChunkSize;

	int nModelsPerCompleteChunk;
	if (!draw_k_from_n(nModelsPerCompleteChunk,
		nPartsForOneModel,
		nEffectiveChunkSize)) {
		return false;
	}
	int nModelsForLastChunk = 0;
	if (!draw_k_from_n(nModelsForLastChunk,
		nPartsForOneModel,
		nPartsOfLastIncompleteChunk)) {
		return false;
	}
	int nExpectedModels = nCompleteChunks * nModelsPerCompleteChunk +
		nModelsForLastChunk;
	if (!nExpectedModels) {
		printf("nExpectedModels==0\n");
		return false;
	}
	int* creationIndices = new int[nExpectedModels*nPartsForOneModel];
	int nModels = 0;
	int nCurModels;
	int* curCreationIndices = create_draw_index_set(nCurModels,
		nPartsForOneModel,
		nEffectiveChunkSize);
	if (!curCreationIndices) {
		printf("failed to create creation index array\n");
		assert(false);
		return false;
	}
	if (!nCurModels) {
		printf("nCurModels==0\n");
		assert(false);
		delete[] curCreationIndices;
		return false;
	}
	for (int i = 0; i < nCompleteChunks; i++) {
		//transfer indices to overall array
		int nCurIndices = nCurModels * nPartsForOneModel;
		int iStartIndex = nModels * nPartsForOneModel;
		int indexShift = i * nEffectiveChunkSize;
		for (int k = 0; k < nCurIndices; k++) {
			creationIndices[iStartIndex + k] = curCreationIndices[k] + indexShift;
		}
		nModels += nCurModels;
	}
	delete[] curCreationIndices;
	if (nModelsForLastChunk > 0) {
		curCreationIndices = create_draw_index_set(nCurModels,
			nPartsForOneModel,
			nPartsOfLastIncompleteChunk);
		if (!curCreationIndices) {
			printf("failed to create creation index array\n");
			assert(false);
			return false;
		}
		if (!nCurModels) {
			printf("nCurModels==0\n");
			assert(false);
			delete[] curCreationIndices;
			return false;
		}
		// Transfer indices to overall array
		int nCurIndices = nCurModels * nPartsForOneModel;
		int iStartIndex = nModels * nPartsForOneModel;
		int indexShift = nCompleteChunks * nEffectiveChunkSize;
		for (int k = 0; k < nCurIndices; k++) {
			creationIndices[iStartIndex + k] = curCreationIndices[k] + indexShift;
		}
		delete[] curCreationIndices;
		nModels += nCurModels;
	}
	assert(nModels == nExpectedModels);
	// Allocate memory for models and create the models
	int nBytesPerModel = pKernel->modelSize();
	char* modelMem = (char*)malloc(nModels*nBytesPerModel);
	bool* modelValid = new bool[nModels];
	char* m = modelMem;
	int* ic = creationIndices;
	for (int i = 0; i < nModels; i++) {
		modelValid[i] = pKernel->construct(m, ic);
		ic += nPartsForOneModel;
		m += nBytesPerModel;
	}
	// Test each model
	int* nConsens = new int[nModels];
	memset(nConsens, 0, sizeof(*nConsens)*nModels);

	for (int i = 0; i < nModels; i++) {
		if (!modelValid[i]) { continue; }
		for (int k = 0; k < nParts; k++) {
			bool bPartWasUsedForCreation = false;
			for (int j = 0; j < nPartsForOneModel; j++) {
				if (k == creationIndices[i*nPartsForOneModel + j]) {
					bPartWasUsedForCreation = true;
				}
			}
			if (bPartWasUsedForCreation) { continue; }
			if (pKernel->accept(modelMem + i * nBytesPerModel, k)) {
				nConsens[i]++;
			}
		}
	}
	int* modelIndex = new int[nModels];
	for (int i = 0; i < nModels; i++) {
		modelIndex[i] = i;
	}
	sort2Arrays(nConsens, modelIndex, nModels, consensus_better);
	// The best model is:
	int iBestModel = modelIndex[0];
	if (modelValid[iBestModel]) {
		//determine again all parts that consens for this model to create the output
		for (int k = 0; k < nParts; k++) {
			if (pKernel->accept(modelMem + iBestModel * nBytesPerModel, k)) {
				cv::DMatch&m = grid_matches[k];
				int ia = m.queryIdx;
				int ib = m.trainIdx;
				//pXC->getIndices(ia,ib,k);
				float v = m.distance;
				/*if (bHoldsVotesOrDistances&&!pXC->getVoteOrDistance(v,k)){
				   deep_err("fields says that it holds votes or distance, but getting them failed for a structure");
				}
				pXR->addRelation(ia,ib,v);*/
				final_matches.push_back(cv::DMatch(ia, ib, v));
			}
		}
		if (pKernel->getTransformation2D(pTransformation2D,
			modelMem + iBestModel * nBytesPerModel)) {
			//pOutputPinXTransformation2D->setData(pTransformation2D);
			//pTransformation2D->Release(_ev_);
		}
	}
	else {
		printf("no valid model was constructed\n");
	}
	for (int i = 0; i < nModels; i++) {
		pKernel->destruct(modelMem + i * nBytesPerModel);
	}
	delete[] modelIndex;
	delete[] nConsens;
	delete[] modelValid;
	free(modelMem);
	delete[] creationIndices;
	return true;
}
/*...END SOURCES FOR GEOMETRIC VERIFICATION*/

/*****************************************************************************/
bool extractNodes_FAST(cv::Mat& img, std::vector<cv::KeyPoint>& v) {
	int threshold = FAST_thresh;
	bool nonmaxSuppression = true;
	cv::FastFeatureDetector detector(threshold, nonmaxSuppression);
	cv::Mat img_8U;
	if (!ensureGrayImageWithDepth(img_8U, CV_8U, img, NULL, false)) {
		cout << "failed to convert image for point detection" << endl;
		return false;
	}
	detector.detect(img_8U, v);
	return true;
}
/*****************************************************************************/
bool extractNodes_SIFT(cv::Mat&img, std::vector<cv::KeyPoint>&v) {
	cv::Mat img_8U;
	if (!ensureGrayImageWithDepth(img_8U, CV_8U, img, NULL, false)) {
		cout << "failed to convert image for point detection" << endl;
		return false;
	}

	cv::SIFT::CommonParams p_common;
	cv::SIFT::DetectorParams p_detector;

	p_common.nOctaves = 4;
	p_common.nOctaveLayers = 3;
	p_common.firstOctave = -1;
	p_common.angleMode = 0;
	p_detector.threshold = 0.04;
	p_detector.edgeThreshold = 10;
	int border = 0;
	cv::SiftFeatureDetector detector(p_detector, p_common);
	if (border == 0) {
		detector.detect(img_8U, v);
	}
	else {
		std::vector<cv::KeyPoint> tmp;
		detector.detect(img, tmp);
		int n = tmp.size();
		v.reserve(n);
		int nLevels = p_common.nOctaves - p_common.firstOctave;
		float*b = new float[nLevels];
		float scalePerLevel = 2.0f;
		for (int i = 0; i < nLevels; i++) {
			b[i] = (float)border*std::pow(scalePerLevel, i);
		}

		float dxb = img.size().width - 1.0f;
		float dyb = img.size().height - 1.0f;
		for (int i = 0; i < n; i++) {
			cv::KeyPoint&k = tmp[i];
			cv::Point2f&p = k.pt;
			int iL = k.octave - p_common.firstOctave;
			float biL = b[iL];
			if (p.x - biL < 0) { continue; }
			if (p.y - biL < 0) { continue; }
			if (p.x + biL > dxb) { continue; }
			if (p.y + biL > dyb) { continue; }
			v.push_back(k);
		}
		delete[] b;
	}

	return true;
}
/*****************************************************************************/
bool extractNodes_DenseSampling(cv::Mat&img, std::vector<cv::KeyPoint>&v) {
	if (!(g_dense_spacing > 0)) {
		printf("invalid spacing\n");
		return false;
	}
	int dx = img.size().width;
	int dy = img.size().height;
	if (dx < 1 || dy < 1) {
		printf("image must have at least 1 pixel\n");
		return false;
	}
	int nX = (int)((float)(dx - 1) / g_dense_spacing) + 1;
	int nY = (int)((float)(dy - 1) / g_dense_spacing) + 1;
	int n = nX * nY;

	float y = 0.0f;
	float var = g_dense_stdDev * g_dense_stdDev;
	int i = 0;
	for (int iy = 0; iy < nY; iy++) {
		float x = 0.0f;
		for (int ix = 0; ix < nX; ix++, i++) {
			if (g_dense_bAddGaussianNoise) {
				float gx = rndNormal((float)x, var);
				float gy = rndNormal((float)y, var);
				// Keep old if outside
				if (gx < 0 || gx >= dx) { gx = x; }
				if (gy < 0 || gy >= dy) { gy = y; }
				v.push_back(cv::KeyPoint(gx, gy, 0));
			}
			else {
				v.push_back(cv::KeyPoint((float)x, (float)y, 0));
			}
			x += g_dense_spacing;
		}
		y += g_dense_spacing;
	}
	return true;
}
/*****************************************************************************/
class ImagePyramid {
public:
	enum ScaleMode { UNDEFINED, DIF_UNIFORM, LOG_UNIFORM };

	// UNIFORM: the difference between sucessive scales is constant
	// LOG_UNIFORM: the fraction between successive scales is constant.
	ImagePyramid() {
		scaleMode = UNDEFINED;
		constScaleDif = 0;
		constScaleFactor = 0;
		bDependingValuesAreValid = false;
		nPyrImages = 0;
		scale_start = 0;
		scale_end = 0;
		relativeScaleRange = 0;
	}
	virtual ~ImagePyramid() {
		for (unsigned int i = 0; i < images.size(); i++) {
			images[i]->release();
		}
		images.clear();
	}
	bool create(cv::Mat*pXI, int nLayers = 4, float lastRelativeScale = 0.125f,
		bool bLogarithmicallyEquallySpaced = false,
		bool bAlwaysScaleFromFirstImage = false) {
		scaleMode = UNDEFINED;
		constScaleDif = 0;
		constScaleFactor = 0;
		cv::Mat*pXCur = pXI;
		pXCur->addref();
		images.push_back(pXCur);
		double scale0 = 1.0;
		scale.push_back(scale0);
		double lastScale = scale0 * lastRelativeScale;
		double q = 0;
		double scaleStep = 0;
		if (nLayers - 1 >= 1) {
			if (bLogarithmicallyEquallySpaced) {
				q = pow((double)lastScale / (double)scale0, 1 / (double)(nLayers - 1));
			}
			scaleStep = (scale0 - lastScale) / (double)(nLayers - 1);
			double curScale = scale0;
			for (int i = 0; i < nLayers - 1; i++) {
				double factor;
				if (bLogarithmicallyEquallySpaced) {
					if (!bAlwaysScaleFromFirstImage) {
						factor = q;
					}
					else {
						factor = pow(q, i + 1);
					}
				}
				else {
					double lastScale = curScale;
					curScale -= scaleStep;
					if (!bAlwaysScaleFromFirstImage) {
						factor = curScale / lastScale;
					}
					else {
						factor = curScale / scale0;
					}
				}
				cv::Mat*pXScaled = new cv::Mat();
				if (bAlwaysScaleFromFirstImage) {
					int dxOrg = pXI->size().width;
					int dyOrg = pXI->size().height;
					int w = ((float)dxOrg)*(float)factor;
					if ((int)(((float)(dxOrg - 1))*factor) == w) {
						w++;
					}
					int h = ((float)dyOrg)*(float)factor;
					cv::resize(*pXI, *pXScaled, cv::Size(w, h), factor, factor, CV_INTER_AREA);
					scale.push_back(factor);
				}
				else {
					cv::resize(*pXCur, *pXScaled, cv::Size(0, 0), factor, factor, CV_INTER_AREA);
					double lastScale = scale.back();
					scale.push_back(lastScale*factor);
				}
				pXCur = pXScaled;
				images.push_back(pXCur);
			}
		}
		if (bLogarithmicallyEquallySpaced) {
			scaleMode = LOG_UNIFORM;
			constScaleFactor = q;
		}
		else {
			scaleMode = DIF_UNIFORM;
			constScaleDif = scaleStep;
		}
		updateDependencies();
		return true;
	}
	bool getBestLevelIndex(int& k, double relativeDownScaleFactor) {
		if (!nPyrImages) { return false; }
		if (nPyrImages == 1) {
			k = 0;
			return true;
		}
		switch (scaleMode) {
		case DIF_UNIFORM:
		{
			double relativeDownScaleRange = 1.0 - relativeDownScaleFactor;
			double fk = ((double)(nPyrImages - 1))*relativeDownScaleRange /
				relativeScaleRange;
			k = (int)fk;
		}
		break;
		case LOG_UNIFORM:
		{
			double fk = log(relativeDownScaleFactor) / logConstScaleFactor;
			k = (int)fk;
		}
		break;
		default:
			return false;
		}
		if (k < 0) { k = 0; }
		if (k > nPyrImages - 1) { k = nPyrImages - 1; }
		return true;
	}
	void updateDependencies() {
		nPyrImages = images.size();
		if (!nPyrImages) {
			bDependingValuesAreValid = false;
			return;
		}
		scale_start = scale[0];
		scale_end = scale.back();
		relativeScaleRange = 1.0f - scale_end / scale_start;
		if (scaleMode == LOG_UNIFORM) {
			logConstScaleFactor = log(constScaleFactor);
		}
		else {
			logConstScaleFactor = 0;
		}
		bDependingValuesAreValid = true;
	}

	// Member variables
	ScaleMode scaleMode;
	double constScaleDif;  // Only meaningful in DIF_UNIFORM mode
	double constScaleFactor;  // Only meaningfull in LOG_UNIFORM mode
	bool bDependingValuesAreValid;
	// Members below are updated using updateDependencies()
	int nPyrImages;  // ==images.size()
	double scale_start;
	double scale_end;
	double relativeScaleRange;
	double logConstScaleFactor;
	vector<cv::Mat*> images;
	vector<double> scale;
};

/*****************************************************************************/
class DList {
public:
	DList() :
		nBuckets(0), nL(0), nPassivePairingsOfBucket(NULL), pairings(NULL) { }
	~DList() {
		if (nPairingsOfBucket) { delete[] nPairingsOfBucket; }
		if (pairings) { delete[] pairings; }
	};

	// nL is the maximum length of a list of a bucket
	bool create(uint64 nBuckets_p, unsigned int nL_p) {
#ifdef _MSV_VER
#pragma warning( push)
#pragma warning( disable : 4127)
#endif
		if (sizeof(size_t) < 8 && nBuckets_p >= 0x100000000ULL) {
			cout << "Must run on a 64-bit machine with current parameter settings"
				<< endl;
			return false;
		}
#ifdef _MSV_VER
#pragma warning( pop)
#endif
		nPairingsOfBucket = new unsigned int[(size_t)nBuckets_p];
		if (!nPairingsOfBucket) {
			cout << "insufficient memory" << endl;
			return false;
		}
		nPassivePairingsOfBucket = new unsigned int[(size_t)nBuckets_p];
		if (!nPassivePairingsOfBucket) {
			cout << "insufficient memory" << endl;
			delete[] nPairingsOfBucket;
			return false;
		}
		memset(nPairingsOfBucket, 0, sizeof(*nPairingsOfBucket)*(size_t)nBuckets_p);
		memset(nPassivePairingsOfBucket, 0,
			sizeof(*nPassivePairingsOfBucket)*(size_t)nBuckets_p);
		pairings = new pair<unsigned int, unsigned int>[(size_t)nBuckets_p*nL_p];
		if (!pairings) {
			cout << "insufficient memory" << endl;
			delete[] nPairingsOfBucket;
			delete[] nPassivePairingsOfBucket;
			nPairingsOfBucket = NULL;
			return false;
		}
		nBuckets = nBuckets_p;
		nL = nL_p;
		return true;
	}

	bool tryInsert(uint64 dtoken, int i, int j) {
		nPassivePairingsOfBucket[dtoken]++;
		if (nPairingsOfBucket[dtoken] >= nL) { return false; }
		pair<unsigned int, unsigned int>&p =
			pairings[dtoken*nL + nPairingsOfBucket[dtoken]++];
		p.first = i;
		p.second = j;
		return true;
	}

	// Returns size of list plus pointer to start via p,
	// plus the number of passive hits to the bucket
	unsigned int getListOfBucket(pair<unsigned int, unsigned int>*& p,
		unsigned int& nPassive, uint64 i) {
		p = pairings + i * nL;
		nPassive = nPassivePairingsOfBucket[i];
		return nPairingsOfBucket[i];
	}

	// Member variables
	uint64 nBuckets;  //number of buckets (= number of possible d-tokens)
	unsigned int nL;

	// nPairingsOfBuckets[i] is the number of pairings contained in bucket i.
	unsigned int* nPairingsOfBucket;

	// nPassivePairingsOfBuckets[i] is the total number of pairings that we
	// tried to insert into bucket i (but were unable to, because of the limited
	// size of each list). It always holds that
	// nPassivePairingsOfBuckets[i] >= nPairingsOfBuckets[i]
	unsigned int* nPassivePairingsOfBucket;

	// pairings[i*nL],...pairings[i*nL+nPairingsOfBucket[i]-1]
	// is the list of pairings of bucket i
	pair<unsigned int, unsigned int>* pairings;
};

/*****************************************************************************/
class Grid {
public:
	Grid(int nS, int nT) : nS(nS), nT(nT) {
		n = nS * nT;
		votes = new float[n];
		memset(votes, 0, sizeof(*votes)*n);
		sourceIndexMapping = new IndexMapInfo[nS];
		tmp = new float[nT];
	}
	~Grid() {
		delete[] tmp;
		delete[] sourceIndexMapping;
		delete[] votes;
	}

	bool vote(DList&s, DList&t) {
		if (s.nBuckets != t.nBuckets) {
			cout << "lists are not compatible" << endl;
			return false;
		}
		cout << endl;
		for (unsigned int i = 0; i < s.nBuckets; i++) {
			pair<unsigned int, unsigned int>* p[2];
			unsigned int nPassivePairs[2];
			unsigned int nPairs[2] = {
			  s.getListOfBucket(p[0],nPassivePairs[0],i),
			  t.getListOfBucket(p[1],nPassivePairs[1],i)
			};
			unsigned int nv = nPassivePairs[0] * nPassivePairs[1];
			if (nv > 0) {
				float voting_power = 1.0f / (float)nv;
				for (unsigned int k = 0; k < nPairs[0]; k++) {
					unsigned int is0 = p[0][k].first;
					unsigned int is1 = p[0][k].second;
					for (unsigned int j = 0; j < nPairs[1]; j++) {
						unsigned int it0 = p[1][j].first;
						unsigned int it1 = p[1][j].second;
						votes[is0*nT + it0] += voting_power;
						votes[is1*nT + it1] += voting_power;
					}
				}
			}
			if (i % (s.nBuckets / 333) == 0 || i == s.nBuckets - 1) {
				double percentage = (double)i / (double)(s.nBuckets - 1);
				cout << '\r' << fixed << showpoint
					<< setprecision(2) << percentage * 100.0 << " % of votes cast";
			}
		}
		return true;
	}

	bool extractCorrespondenceHypotheses(std::vector<cv::DMatch>& matches,
		int qualityMode, int nBest = -1,
		bool bDiscardMultipleHits = true) {
		for (int is = 0; is < nS; is++) {
			memcpy(tmp, votes + is * nT, sizeof(*votes)*nT);
			float power = normalizeVector_noabs(tmp, nT);
			float sum = 0;
			float log2 = log(2.0f);
			int iBest = -1;
			int iBest2 = -1;
			float pbest = 0;
			float pbest2 = 0;
			for (int i = 0; i < nT; i++) {
				float p = tmp[i];
				if (p > pbest) {
					if (iBest != -1) {
						if (p > pbest2) {
							pbest2 = pbest;
							iBest2 = iBest;
						}
					}
					pbest = p;
					iBest = i;
				}
				else if (p > pbest2) {
					pbest2 = p;
					iBest2 = i;
				}
				if (p != 0) {
					// See http://en.wikipedia.org/wiki/Entropy_%28information_theory%29
					// p*log(p) is taken to be zero, which is consistent with
					// the limit \limes_{p->0+}{p log(p)}=0
					sum += p * log(p);
				}
			}
			IndexMapInfo&m = sourceIndexMapping[is];
			m.index = is;
			m.entropy = -sum / log2;
			m.i1stBestMatch = iBest;
			m.i2ndBestMatch = iBest2;
			m.p1 = pbest;
			m.p2 = pbest2;
			m.power = power;
			switch (qualityMode) {
			case 0:
				if (m.entropy != 0) {
					m.quality = m.power*m.p1 / m.entropy;
				}
				else {
					m.quality = m.power*m.p1 / 0.1f;
				}
				break;
			case 1:
				m.quality = m.power*m.p1;
				break;
			case 2:
				m.quality = -m.entropy;
				break;
			case 3:
				m.quality = m.entropy;
				break;
			case 4:
				m.quality = m.p1;
				break;
			case 5:
				if (pbest2 != 0) {
					m.quality = m.p1 / m.p2;
				}
				else {
					m.quality = 0;
				}
				break;
			case 6:
				m.quality = m.p1 - m.p2;
				break;
			default:
				m.quality = m.p1;
			}
		}
		qsort(sourceIndexMapping, nS, sizeof(*sourceIndexMapping),
			IndexMapInfo_Quality_Cmp);
		bool* covered = NULL;
		if (bDiscardMultipleHits) {
			covered = new bool[nT];
			memset(covered, 0, sizeof(*covered)*nT);
		}
		int count = 0;
		for (int i = 0; i < nS; i++) {
			IndexMapInfo&m = sourceIndexMapping[i];
			if (m.i1stBestMatch != -1) {
				count++;
				if (nBest != -1 && count >= nBest) { break; }
				if (bDiscardMultipleHits&&covered[m.i1stBestMatch]) { continue; }
				matches.push_back(cv::DMatch(m.index, m.i1stBestMatch, 1.0f / m.quality));
				if (bDiscardMultipleHits) { covered[m.i1stBestMatch] = true; }
			}
		}
		if (covered) { delete[] covered; }
		return true;
	}

	bool visualize() {
		if (!votes) { return false; }
		float min, max;
		if (!getMinAndMaxOfVector(min, max, votes, n)) { return false; }
		cv::Mat I(nS, nT, CV_32FC1, votes);
		cv::Mat I2;
		I.convertTo(I2, CV_32FC1, 1.0f / (max - min), -min);
		imshow("Voting Grid", I2);
		return true;
	}

	IndexMapInfo*sourceIndexMapping;

	// Member variables
	// Consecutive values in memory correspond to to the same source node,
	// but to different target nodes
	float*votes;
	float*tmp;
	int nS;  // number of nodes in image A (source)
	int nT;  // number of nodes in image B (target)
	int n;   // number of cells
};


/*****************************************************************************/

bool pix(float& f, float x, float y, float* data, cv::Size& s) {
#define px(xx,yy) data[(yy)*s.width+xx]
	int iy = (int)y;
	int ix = (int)x;
	if (iy < 0 || iy >= s.height || ix < 0 || ix >= s.width) { return false; }
	if (y >= s.height) { y = (float)s.height - 1; }
	if (x >= s.width) { x = (float)s.width - 1; }
	float r = 1.0f - (y - (float)iy);
	float c = 1.0f - (x - (float)ix);
	float omc = 1.0f - c;
	float omr = 1.0f - r;
	float px_ixiy = px(ix, iy);
	float y1 = 0;
	float y2 = 0;
	y1 = c < 1 ? c * px_ixiy + omc * px(ix + 1, iy) : px_ixiy;
	if (r < 1) {
		y2 = c < 1 ? c * px(ix, iy + 1) + omc * px(ix + 1, iy + 1) : px(ix, iy + 1);
	}
	f = r * y1 + omr * y2;
	return true;
}

/*****************************************************************************/
bool getStrip(bool&bAllValid, float*f, int nF,
	float ax, float ay, float bx, float by, float*data, cv::Size s) {
	if (!f) { return false; }
	if (nF <= 0) { return false; }
	float vx = bx - ax;
	float vy = by - ay;
	float px = ax;
	float py = ay;
	float vstepx, vstepy;
	if (nF == 1) {  //sample center (if nF==1)
		px = ax + vx * 0.5f;
		py = by + vy * 0.5f;
		vstepx = 0;
		vstepy = 0;
	}
	else {
		float factor = 1.0f / (float)(nF - 1);
		vstepx = vx * factor;
		vstepy = vy * factor;
	}
	bAllValid = true;
	for (int i = 0; i < nF; i++) {
		if (!pix(f[i], px, py, data, s)) {
			bAllValid = false;
			return false;
		}
		px += vstepx;
		py += vstepy;
	}
	return true;
}

///////////////////////////////////////////////////////////////////////////////
template <class REAL> void resizeVectorByAveraging(REAL*dst, int nDst, REAL*src,
	int nSrc, REAL*countBuf) {
	memset(dst, 0, sizeof(*dst)*nDst);
	memset(countBuf, 0, sizeof(*countBuf)*nDst);
	if (nSrc >= nDst) {
		double indexStep = (double)(nDst - 1) / (double)(nSrc - 1);
		double k = 0;
		for (int i = 0; i < nSrc; i++) {
			int ik = (int)(k + 0.001);
			dst[ik] += src[i];
			countBuf[ik] += 1.0;
			k += indexStep;
		}
	}
	else {
		double indexStep = (double)(nSrc - 1) / (double)(nDst - 1);
		double k = 0;
		for (int i = 0; i < nDst; i++) {
			int ik = (int)(k + 0.001);
			dst[i] += src[ik];
			countBuf[i] += 1.0;
			k += indexStep;
		}
	}
	for (int i = 0; i < nDst; i++) {
		dst[i] /= countBuf[i];
	}
}


/*****************************************************************************/
bool extract_dtoken(uint64& dtoken, ImagePyramid& pyr,
	cv::KeyPoint& a, cv::KeyPoint&b,
	float* tmpIntensities, float* tmpCount, float* avg) {
	// Determine which pyramid layer to use, depeding on the length of line [ab]
	float dxab = b.pt.x - a.pt.x;
	float dyab = b.pt.y - a.pt.y;
	float l2ab = dxab * dxab + dyab * dyab;
	float lab = sqrt(l2ab);
	float relativeDownScaleFactor = ((float)nSections*pyramidAccessFactor) / lab;
	int k;
	if (!pyr.getBestLevelIndex(k, relativeDownScaleFactor)) {
		cout << "failed to access image pyramid\n";
		return false;
	}

	cv::Mat*pXI = pyr.images[k];
	float*data = (float*)pXI->data;
	cv::Size s = pXI->size();
	//project point a and b onto the respective pyramid level
	float scale = (float)pyr.scale[k];
	float ax_pyr = a.pt.x*scale;
	float ay_pyr = a.pt.y*scale;
	float bx_pyr = b.pt.x*scale;
	float by_pyr = b.pt.y*scale;
	float l_pyr = lab * scale;
	float vx_pyr = bx_pyr - ax_pyr;
	float vy_pyr = by_pyr - ay_pyr;
	float sx = ax_pyr + vx_pyr * q0;
	float sy = ay_pyr + vy_pyr * q0;
	float ex = ax_pyr + vx_pyr * q1;
	float ey = ay_pyr + vy_pyr * q1;
	int nSamples = (int)l_pyr;
	if (nSamples < nSections) { nSamples = nSections; }
	bool bAllValid;
	if (!getStrip(bAllValid, tmpIntensities, nSamples, sx, sy, ex, ey, data, s)) {
		cout << "failed to get strip\n";
		return false;
	}
	resizeVectorByAveraging(avg, nSections, tmpIntensities, nSamples, tmpCount);
	if (!normalizeVectorMinMaxZeroOne(avg, nSections)) {
		for (int i = 0; i < nSections; i++) {
			avg[i] = 0.5f;
		}
	}

	// Quantize each section by descretizing its normalized average
	// intensity and concatenate the bits
	dtoken = 0;
	for (int i = 0; i < nSections; i++) {
		int value = (int)((avg[i] * floatNValuesPerSubSection));
		if (value < 0) { value = 0; }
		if (value >= nValuesPerSubSection) { value = nValuesPerSubSection - 1; }
		dtoken <<= bitsPerSection;
		dtoken |= value;
	}
	return true;
}
/*****************************************************************************/
class KeyPointLinkAtUniquePos : public CvPoint2D32f {
public:
	KeyPointLinkAtUniquePos(float x_p, float y_p,
		int idx, KeyPointLinkAtUniquePos*next = NULL) :
		idx(idx), next(next) {
		x = x_p; y = y_p;
	}
	int idx;//index of original keypoin
	class KeyPointLinkAtUniquePos* next;
};
/*****************************************************************************/
bool total_order_2D(cv::KeyPoint& a, cv::KeyPoint& b) {
	if (a.pt.y < b.pt.y) { return true; }
	if (a.pt.y > b.pt.y) { return false; }
	if (a.pt.x < b.pt.x) { return true; }
	if (a.pt.x > b.pt.x) { return false; }
	return true;
}
/*****************************************************************************/
void getMappingOfIndex(std::vector<int>& ibs,
	const vector<cv::DMatch>& matches1to2, int ia) {
	for (int i = 0; i < matches1to2.size(); i++) {
		if (matches1to2[i].queryIdx == ia) {
			ibs.push_back(matches1to2[i].trainIdx);
		}
	}
}
/*****************************************************************************/
class Mesh {
public:
	Mesh() { bDirectIndicesToOrgPoints = true; }
	Mesh(CvRect imageRect, const vector<cv::KeyPoint>& keypoints_p,
		const vector<cv::DMatch>& matches1to2) : imageRect(imageRect) {
		bDirectIndicesToOrgPoints = false;
		org_keypoints = keypoints_p;
		create_unique_point_lists(keypoints_p, matches1to2);
		triangulate();
	}
	void create_unique_point_lists(const vector<cv::KeyPoint>& keypoints_p,
		const vector<cv::DMatch>& matches1to2) {
		/*
			 * Triangulation methods typically have problems with multiple points
		 * at identical locations. Hence, as a preprocessing step, created
		 * linked lists of points with identical locations, such that each list,
		 * possibly with only 1 element, can be considered as a unique point
		 * for triangulation.
			 * Edges than apply for all members of the linked lists of the source
		 * and target of an edge, respectively.
		 */
		vector<int> idx;  // To keep track of index permutations
		for (int i = 0; i < keypoints_p.size(); i++) {  //make a local copy first
			std::vector<int> map_i;
			getMappingOfIndex(map_i, matches1to2, i);
			if (map_i.size()) {
				idx.push_back(i);
			}
		}
		int n = idx.size();
		cv::KeyPoint* keypoints = new cv::KeyPoint[n];
		for (int i = 0; i < n; i++) {
			keypoints[i] = keypoints_p[idx[i]];
		}

		// Sorting them with total_order_2d will let points with identical
	// coordinates be at subsequent positions after sorting.
		sort2Arrays(keypoints, (int*)idx.data(),
			n, total_order_2D);

		unique_keypoint_lists.reserve(n);  // upper bound
		for (int i = 0; i < n; i++) {
			float xroot = keypoints[i].pt.x;
			float yroot = keypoints[i].pt.y;
			KeyPointLinkAtUniquePos* pRoot =
				new KeyPointLinkAtUniquePos(xroot, yroot, idx[i]);
			KeyPointLinkAtUniquePos* pTail = pRoot;
			unique_keypoint_lists.push_back(pRoot);
			i++;
			while (i < n) {
				float xnext = keypoints[i].pt.x;
				float ynext = keypoints[i].pt.y;
				if (xnext != xroot || ynext != yroot) { break; }
				KeyPointLinkAtUniquePos* pNext =
					new KeyPointLinkAtUniquePos(xnext, ynext, idx[i], pTail);
				pTail = pNext;
				unique_keypoint_lists.push_back(pNext);
				i++;
			}
		}
		delete[] keypoints;
	}
	void triangulate(bool bBiDirectional = false) {
		int nP = unique_keypoint_lists.size();
		V2* p = new V2[nP];
		for (int i = 0; i < nP; i++) {
			p[i].x = unique_keypoint_lists[i]->x;
			p[i].y = unique_keypoint_lists[i]->y;
		}
		Triangulator d2(false, true, p, nP);
		fx_set<R2> tmp_edges;
		for (int i = 0; i < d2.final_triangles.size(); i++) {
			I3& i3 = d2.final_triangles[i];
			if (bBiDirectional) {
				tmp_edges.insert(R2(i3.ia, i3.ib));
				tmp_edges.insert(R2(i3.ib, i3.ia));
				tmp_edges.insert(R2(i3.ia, i3.ic));
				tmp_edges.insert(R2(i3.ic, i3.ia));
				tmp_edges.insert(R2(i3.ib, i3.ic));
				tmp_edges.insert(R2(i3.ic, i3.ib));
			}
			else {
				if (!tmp_edges.contains(R2(i3.ia, i3.ib))) {
					tmp_edges.insert(R2(i3.ib, i3.ia));
				}
				if (!tmp_edges.contains(R2(i3.ia, i3.ic))) {
					tmp_edges.insert(R2(i3.ic, i3.ia));
				}
				if (!tmp_edges.contains(R2(i3.ib, i3.ic))) {
					tmp_edges.insert(R2(i3.ic, i3.ib));
				}
			}
		}
		for (fx_set<R2>::iterator i = tmp_edges.begin();
			i != tmp_edges.end();
			i++) {
			edges.push_back(*i);
		}
		delete[] p;
	}
	void transfer(Mesh&m2, const vector<cv::DMatch>& matches1to2,
		const vector<cv::KeyPoint>& keypoints2) {
		m2.org_keypoints = keypoints2;
		//for each edge in this mesh
		//use the mapping matches1to2 to map the indices of the points
		//and add a respective edges to mesh m2
		fx_set<R2> tmp_edges; //for m2
		for (int i = 0; i < edges.size(); i++) {
			int iLA = edges[i].ia;
			int iLB = edges[i].ib;
			KeyPointLinkAtUniquePos* pa = unique_keypoint_lists[iLA];
			while (pa) {
				KeyPointLinkAtUniquePos* pb = unique_keypoint_lists[iLB];
				while (pb) {
					//ids of the original keypoints...
					int idxa = pa->idx;
					int idxb = pb->idx;
					std::vector<int> idxa_maps;
					std::vector<int> idxb_maps;
					getMappingOfIndex(idxa_maps, matches1to2, idxa);
					getMappingOfIndex(idxb_maps, matches1to2, idxb);
					for (int k = 0; k < idxa_maps.size(); k++) {
						int ia_cur_map = idxa_maps[k];
						for (int j = 0; j < idxb_maps.size(); j++) {
							int ib_cur_map = idxb_maps[j];
							tmp_edges.insert(R2(ia_cur_map, ib_cur_map));
						}
					}
					pb = pb->next;
				}
				pa = pa->next;
			}
		}
		for (fx_set<R2>::iterator i = tmp_edges.begin();
			i != tmp_edges.end();
			i++) {
			m2.edges.push_back(*i);
		}
	}

	void getEdgeIndices(std::vector<R2>&edges_p) {
		//returned indices point into unique_keypoint_lists
		edges_p = edges;
	}

	void drawIntoImage(cv::Mat&outImg, CvScalar color,
		int thickness, int lineType, int shift) {
		std::vector<R2 > edges;
		getEdgeIndices(edges);
		const int draw_multiplier = 1 << shift;
		for (int i = 0; i < edges.size(); i++) {
			cv::Point2f a;
			cv::Point2f b;
			if (bDirectIndicesToOrgPoints) {
				int ia = edges[i].ia;
				int ib = edges[i].ib;
				a.x = org_keypoints[ia].pt.x*draw_multiplier;
				a.y = org_keypoints[ia].pt.y*draw_multiplier;
				b.x = org_keypoints[ib].pt.x*draw_multiplier;
				b.y = org_keypoints[ib].pt.y*draw_multiplier;
			}
			else {
				int ila = edges[i].ia;
				int ilb = edges[i].ib;
				KeyPointLinkAtUniquePos* pLA = unique_keypoint_lists[ila];
				KeyPointLinkAtUniquePos* pLB = unique_keypoint_lists[ilb];
				a = cv::Point2f(pLA->x*draw_multiplier, pLA->y*draw_multiplier);
				b = cv::Point2f(pLB->x*draw_multiplier, pLB->y*draw_multiplier);
			}
			line(outImg, a, b, color, thickness, lineType, shift);
		}
		/*Point2f pt1 = kp1.pt,
				pt2 = kp2.pt,
				dpt2 = Point2f( std::min(pt2.x+outImg1.cols, float(outImg.cols-1)), pt2.y );

		line( outImg,
			  Point(cvRound(pt1.x*draw_multiplier), cvRound(pt1.y*draw_multiplier)),
			  Point(cvRound(dpt2.x*draw_multiplier), cvRound(dpt2.y*draw_multiplier)),
			  color, 1, CV_AA, draw_shift_bits );*/
	}
	virtual ~Mesh() {
		for (unsigned int i = 0; i < unique_keypoint_lists.size(); i++) {
			KeyPointLinkAtUniquePos* tmp = unique_keypoint_lists[i];
			KeyPointLinkAtUniquePos* nxt;
			do {
				nxt = tmp->next;
				delete tmp;
				tmp = nxt;
			} while (tmp);
			unique_keypoint_lists.clear();
		}
	}
protected:
	CvRect imageRect;
	vector<cv::KeyPoint> org_keypoints;
	std::vector<KeyPointLinkAtUniquePos*> unique_keypoint_lists;
	std::vector<R2> edges;  // Indexes into unique_keypoint_lists or
							// org_keypoints
				// (if bDirectIndicesToOrgPoints is true)
	bool bDirectIndicesToOrgPoints;
};
/*****************************************************************************/
void prepareImgAndDrawKeypointsForMeshes(const cv::Mat& img1,
	const vector<cv::KeyPoint>& keypoints1,
	const cv::Mat& img2,
	const vector<cv::KeyPoint>& keypoints2,
	cv::Mat& outImg,
	cv::Mat& outImg1,
	cv::Mat& outImg2,
	const cv::Scalar& singlePointColor1,
	const cv::Scalar& singlePointColor2,
	int flags) {
	cv::Size size(img1.cols + img2.cols, MAX(img1.rows, img2.rows));
	if (flags & cv::DrawMatchesFlags::DRAW_OVER_OUTIMG) {
		if (size.width > outImg.cols || size.height > outImg.rows) {
			CV_Error(CV_StsBadSize,
				"outImg size less than needed to draw img1 and img2");
		}
		outImg1 = outImg(cv::Rect(0, 0, img1.cols, img1.rows));
		outImg2 = outImg(cv::Rect(img1.cols, 0, img2.cols, img2.rows));
	}
	else {
		outImg.create(size, CV_MAKETYPE(img1.depth(), 3));
		outImg1 = outImg(cv::Rect(0, 0, img1.cols, img1.rows));
		outImg2 = outImg(cv::Rect(img1.cols, 0, img2.cols, img2.rows));
		if (img1.type() == CV_8U) {
			cvtColor(img1, outImg1, CV_GRAY2BGR);
		}
		else {
			img1.copyTo(outImg1);
		}
		if (img2.type() == CV_8U) {
			cvtColor(img2, outImg2, CV_GRAY2BGR);
		}
		else {
			img2.copyTo(outImg2);
		}
	}

	// draw keypoints
	if (!(flags & cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS)) {
		cv::Mat outImg1 = outImg(cv::Rect(0, 0, img1.cols, img1.rows));
		drawKeypoints(outImg1, keypoints1, outImg1, singlePointColor1,
			flags | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
		cv::Mat outImg2 = outImg(cv::Rect(img1.cols, 0, img2.cols, img2.rows));
		drawKeypoints(outImg2, keypoints2, outImg2, singlePointColor2,
			flags | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
	}
}

/*****************************************************************************/
static inline void drawKeypoint_xx(cv::Mat& img, const cv::KeyPoint& p,
	const cv::Scalar& color, int flags,
	int draw_shift_bits) {
	int draw_multiplier = 1 << draw_shift_bits;
	CV_Assert(!img.empty());
	cv::Point center(cvRound(p.pt.x * draw_multiplier),
		cvRound(p.pt.y * draw_multiplier));
	// draw center with R=3
	int radius = 1 * draw_multiplier;
	cv::circle(img, center, radius, color, 1, CV_AA, draw_shift_bits);
}
/*****************************************************************************/
CvPoint2D32f CrossPoint1(cv::KeyPoint p1, cv::KeyPoint p2, cv::KeyPoint p3, cv::KeyPoint p4)  //计算两条直线的交点。直线由整数向量形式提供。
{
	CvPoint2D32f pt;           double k1, k2, b1, b2;
	if (p1.pt.x == p2.pt.x)//如果第一条直线斜率不存在
	{
		pt.x = p1.pt.x;
		pt.y = p3.pt.y == p4.pt.y ? p3.pt.y :
			double(p3.pt.y - p4.pt.y)*(pt.x - p3.pt.x) / (p3.pt.x - p4.pt.x) + p3.pt.y;
	}
	else if (p3.pt.x == p4.pt.x)//如果第二条直线斜率不存在
	{
		pt.x = p3.pt.x;
		pt.y = p1.pt.y == p2.pt.y ? p1.pt.y :
			double(p1.pt.y - p2.pt.y)*(pt.x - p1.pt.x) / (p1.pt.x - p2.pt.x) + p1.pt.y;
	}
	else     //求出斜截式方程。然后让k1x + b1 = k2x + b2，解出x，再算出y即可
	{
		k1 = double(p2.pt.y - p1.pt.y) / (p2.pt.x - p1.pt.x);      b1 = double(p1.pt.y - k1 * p1.pt.x);
		k2 = double(p4.pt.y - p3.pt.y) / (p4.pt.x - p3.pt.x);      b2 = double(p3.pt.y - k2 * p3.pt.x);
		pt.x = (b2 - b1) / (k1 - k2);  //算出x
		pt.y = k1 * pt.x + b1; //算出y
	}
	return pt;
}
/*****************************************************************************/
void drawMatchesThroughMeshes(const cv::Mat& img1,
	const vector<cv::KeyPoint>& keypoints1,
	const cv::Mat& img2,
	const vector<cv::KeyPoint>& keypoints2,
	const vector<cv::DMatch>& matches1to2,
	cv::Mat& outImg,
	const cv::Scalar& matchColor = cv::Scalar::all(-1),
	const cv::Scalar& singlePointColor1 = cv::Scalar::all(-1),
	const cv::Scalar& singlePointColor2 = cv::Scalar::all(-1),
	const vector<char>& matchesMask = vector<char>(),
	int flags = cv::DrawMatchesFlags::DEFAULT) {
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 1);

	if (!matchesMask.empty() && matchesMask.size() != matches1to2.size()) {
		CV_Error(CV_StsBadSize, "matchesMask must have the same size as matches1to2");
	}
	cv::Mat outImg1, outImg2;
	prepareImgAndDrawKeypointsForMeshes(img1, keypoints1, img2, keypoints2,
		outImg, outImg1, outImg2,
		singlePointColor1, singlePointColor2,
		flags);
	//drawMatches(img1,keypoints1,img2,keypoints2,matches1to2,outImg);

	CvRect r1;
	r1.x = 0;
	r1.y = 0;
	r1.width = img1.size().width;
	r1.height = img1.size().height;
	Mesh m1(r1, keypoints1, matches1to2);
	CvRect r2;
	r2.x = 0;
	r2.y = 0;
	r2.width = img2.size().width;
	r2.height = img2.size().height;

	Mesh m2;
	m1.transfer(m2, matches1to2, keypoints2);
	int lineWidth = g_displayMeshLineWidth;
	m1.drawIntoImage(outImg1, color1, lineWidth, CV_AA, 4);
	m2.drawIntoImage(outImg2, color2, lineWidth, CV_AA, 4);
	IplImage pic1, pic2;


	cv::RNG& rng = cv::theRNG();
	bool isRandMatchColor = matchColor == cv::Scalar::all(-1);

	for (int i = 0; i < matches1to2.size(); i++) {
		cv::Scalar color = isRandMatchColor ?
			cv::Scalar(rng(256), rng(256), rng(256)) :
			matchColor;
		const cv::KeyPoint&kp1 = keypoints1[matches1to2[i].queryIdx];
		const cv::KeyPoint&kp2 = keypoints2[matches1to2[i].trainIdx];
		char text1[100];
		sprintf(text1, "%d", matches1to2[i].queryIdx);
		char text2[100];
		sprintf(text2, "%d", matches1to2[i].trainIdx);
		pic1 = IplImage(img1);
		pic2 = IplImage(img2);
		drawKeypoint_xx(outImg1, kp1, color, flags, 4);
		cvPutText(&pic1, text1, kp1.pt, &font, CV_RGB(255, 0, 0));
		drawKeypoint_xx(outImg2, kp2, color, flags, 4);
		cvPutText(&pic2, text2, kp2.pt, &font, CV_RGB(255, 0, 0));
		cout << "WHAT" << endl;
		cvShowImage("1", &pic1);
		cvShowImage("2", &pic2);

	}
}
/*****************************************************************************/
void help() {
	cout << "\
This program demonstrates D-Nets for image matching. The code implements  \n\
the exhausitve version (Clique D-Nets) of our CVPR2012 paper:             \n\
                                                                          \n\
              D-Nets: Beyond Patch-Based Image Descriptors                \n\
                                                                          \n\
              Felix von Hundelshausen       Rahul Sukthankar              \n\
              felix.v.hundelshausen@live.de rahuls@cs.cmu.edu             \n\
                                                                          \n\
 IEEE International Conference on Computer Vision and Pattern Recognition \n\
              June 18-20, 2012, Providence, Rhode Island, USA             \n\
                                                                          \n\
                                                                          \n\
The program matches two images using FAST interest points as nodes.       \n\
                                                                          \n\
We recommend running the program on a 64-bit architecture with 16 GB      \n\
memory. If the program is executed on a 32-bit architecture, or with      \n\
much less memory, set parameter s<=10 (for b=2) through the command       \n\
line parameters. However the difficult matching cases with a large        \n\
scale change and many interest points will require values up to s=13.     \n\
Those cases can only be sucessfully run on a 64-bit machine with enough   \n\
memory.                                                                   \n\
Command Line Parameters                                                   \n\
-----------------------                                                   \n\
The first 4 command line parameters need to be supplied always and in a   \n\
fixed order, e.g.,:                                                       \n\
                1           2          3          4                       \n\
           -------------- ----- --------------- -----                     \n\
   ./dnets boats/img1.pgm s=1.0 boats/img2.pgm  w=640                     \n\
                                                                          \n\
The first 4 command line parameters consist of two groups of 2 parameters.\n\
Within each group, the first parameter is a filename of an input image,   \n\
the second parameter determines how the respective image should be scaled \n\
initially.                                                                \n\
Conventional image formats (such as *.png, *.jpg, *.bmp) can be read.     \n\
But color images will be converted to grayscale images internally.        \n\
The initial scaling can be specified in two ways:                         \n\
    1. The first alternative,  e.g. \"s=0.5\" specified that the image    \n\
       is to be scaled to have a resulting scale of 50% of the original   \n\
       image (in this example). That is, to leave the original image      \n\
       unscaled \"s=1\" can be specified.                                 \n\
    2. The second alternative, e.g. \"w=640\" specifies that the image    \n\
       should be scaled, such that after scaling it, the width should be  \n\
       640 pixels (in this example).                                      \n\
                                                                          \n\
Some examples for valid calls with the first 4 parameters are:            \n\
   ./dnets boats/img1.png s=1 /boats/img2.png s=1                         \n\
   ./dnets boats/img1.png w=800 /boats/img2.png s=0.4                     \n\
   ./dnets boats/img1.png w=300 /boats/img2.png w=640                     \n\
That is, the flag \"s=\" or \"w=\" always relates to the preceeding image.\n\
The fist example will leave both images at their original scale.          \n\
After the first 4 parameters in fixed order, there can be further optional\n\
parameters with no required specific order.                               \n\
They are:                                                                 \n\
   sigma=[1.0]        Smooth the initial image with a Gaussian kernel with\n\
                      standard deviation 1.0.                             \n\
   kx=[FAST]          Choose FAST as node extractor                       \n\
     =SIFT            Choose SIFT as node extractor                       \n\
     =DENSE_SAMPLING  Choose DENSE_SAMPLING ad node extractor             \n\
   ds_spacing=[10]    Spacing of points in dense sampling                 \n\
   gm=NONE            Disable geometric verification                      \n\
     =AFFINE          Affine geometric verification                       \n\
     =[HOMOGRAPHY]    Homography geometric verification                   \n\
   gv_dist=[16]       Distance for geometrice verification (if enabled)   \n\
   FAST_thresh=[80]   Use a threshold of 80 for FAST-keypoint extraction. \n\
   L=[8]              Create an image pyramid with 8 levels.              \n\
   q0=0.1             Encode a strip starting at 0.1% of the strip.       \n\
   q1=0.8             Encode strip up to 0.8% of the strip.               \n\
   nS=[9]             Divide a strip into nS sections                     \n\
   b=[2]              Use 2 bits to encode each section.                  \n\
   nL=[20]            Limit the size of the lists in the hash table to 20 \n\
                      pairings.                                           \n\
   om=matches.jpg     Create file \"matches.jpg\" showing the matches     \n\
                      between both images based on OpenCV's drawMatches   \n\
                      fuction.                                            \n\
   nM=40              Only extract the best 40 matches. If not supplied   \n\
                      all matches are extractd by default.                \n\
   vis=[LINES]        Visualize final correspondences using lines.       \n\
      =MESHES         Visualize the correspondences via meshes.           \n\
   wait=false         Do not wait for ESC at the end but exit immediately.\n\
                                                                          \n\
   -h                 print this command line info and exit.              \n\
   --h                print this command line info and exit.              \n\
   (defaults are indicated by [] but the braces are not part of the input)\n\
                                                                          \n\
   Calling dnets without any argmuents will print this help, too.\n";
}

/*****************************************************************************/
int main(int argc, char** argv) {
	if (argc < 5) {
		help();
		return 0;
	}

	init();
	cv::Mat img[2];
	ImagePyramid pyr[2];

	// Read and initialize images based on the first 4 command line parameters
	if (!readAndConvertImages(img, argc, argv)) { return 0; }

	//process optional command line parameters...
	for (int i = 5; i < argc; i++) {
		string flag;
		float value;
		string param;
		bool bFloatValid;
		if (readParam(flag, param, bFloatValid, value, argv[i])) {
			if (flag == "sigma"&&bFloatValid) {
				sigma = value;
				printf("sigma=%f\n", sigma);
			}
			else if (flag == "vis" && !strcmp(param.c_str(), "LINES")) {
				gVisualizationType = LINES;
				printf("gVisualizationType=LINES\n");
			}
			else if (flag == "vis" && !strcmp(param.c_str(), "MESHES")) {
				gVisualizationType = MESHES;
				printf("gVisualizationType=MESHES\n");
			}
			else if (flag == "kx" && !strcmp(param.c_str(), "DENSE_SAMPLING")) {
				gKeypointExtractionType = DENSE_SAMPLING;
				printf("gKeypointExtractionType=DENSE_SAMPLING\n");
			}
			else if (flag == "kx" && !strcmp(param.c_str(), "FAST")) {
				gKeypointExtractionType = FAST;
				printf("gKeypointExtractionType=FAST\n");
			}
			else if (flag == "kx" && !strcmp(param.c_str(), "SIFT")) {
				gKeypointExtractionType = SIFT;
				printf("gKeypointExtractionType=SIFT\n");
			}
			else if (flag == "ds_spacing"&&bFloatValid) {
				g_dense_spacing = (int)value;
				printf("g_dense_spacing=%d\n", g_dense_spacing);
			}
			else if (flag == "gm" && !strcmp(param.c_str(), "NONE")) {
				gTransformationModel = NONE;
				printf("gTransformationModel=NONE\n");
			}
			else if (flag == "gm" && !strcmp(param.c_str(), "AFFINE")) {
				gTransformationModel = AFFINE;
				printf("gTransformationModel=AFFINE\n");
			}
			else if (flag == "gm" && !strcmp(param.c_str(), "HOMOGRAPHY")) {
				gTransformationModel = HOMOGRAPHY;
				printf("gTransformationModel=HOMOGRAPHY\n");
			}
			else if (flag == "gv_dist"&&bFloatValid) {
				g_geometricVerification_threshDist = value;
				printf("g_geometricVerification_threshDist=%f\n",
					g_geometricVerification_threshDist);
			}
			else if (flag == "FAST_thresh"&&bFloatValid) {
				FAST_thresh = (int)value;
				printf("FAST_thresh=%d\n", FAST_thresh);
			}
			else if (flag == "L"&&bFloatValid) {
				nLayers = (int)value;
				printf("nLayers=%d\n", nLayers);
			}
			else if (flag == "q0"&&bFloatValid) {
				q0 = value;
				printf("q0=%f\n", q0);
			}
			else if (flag == "q1"&&bFloatValid) {
				q1 = value;
				printf("q1=%f\n", q1);
			}
			else if (flag == "nS"&&bFloatValid) {
				nSections = (int)value;
				printf("nSections=%d\n", nSections);
			}
			else if (flag == "b"&&bFloatValid) {
				bitsPerSection = (int)value;
				printf("bitsPerSection=%d\n", bitsPerSection);
			}
			else if (flag == "nL"&&bFloatValid) {
				nL = (int)value;
				printf("nL=%d\n", nL);
			}
			else if (flag == "om") {
				outputFilename_VisualizedMatches = param;
				printf("outputFilename=%s\n", outputFilename_VisualizedMatches.c_str());
			}
			else if (flag == "nM"&&bFloatValid) {
				nExtractOnlyNBest = (int)value;
				printf("nExtractOnlyNBest=%d\n", nExtractOnlyNBest);
			}
			else if (flag == "wait"&&param == "false") {
				bWait = false;
			}
			else if (flag == "h") {
				help();
				return 0;
			}
		}
	}
	//计算程序运行时间
	double dur;
	clock_t start, end;
	start = clock();

	// Show input images... (after initial scaling operations and conversion
	// to a float image)
	imshow("image 0", img[0]);
	imshow("image 1", img[1]);

	// Compute some values...
	nMaxDTokens = 1 << (nSections*bitsPerSection);
	nValuesPerSubSection = 1 << bitsPerSection;
	floatNValuesPerSubSection = (float)nValuesPerSubSection;

	cout << endl;
	//extract points as nodes
	std::vector<cv::KeyPoint> v[2];
	for (int i = 0; i < 2; i++) {
		switch (gKeypointExtractionType) {
		case FAST:
			if (!extractNodes_FAST(img[i], v[i])) { return 0; }
			break;
		case DENSE_SAMPLING:
			if (!extractNodes_DenseSampling(img[i], v[i])) { return 0; }
			break;
		case SIFT:
			if (!extractNodes_SIFT(img[i], v[i])) { return 0; }
			break;
		}
		cout << v[i].size() << " interest points extracted\n";
	}
	cv::Mat smoothed_img[2];
	// Smooth images
	if (sigma > 0) {
		int kernel_size = (int)max(3.0f, 8 * sigma + 1.0f);
		for (int i = 0; i < 2; i++) {
			GaussianBlur(img[i], smoothed_img[i],
				cv::Size(kernel_size, kernel_size),
				sigma, sigma);
		}
	}

	for (int i = 0; i < 2; i++) {
		pyr[i].create(&smoothed_img[i], nLayers, 1.0f / (float)nLayers, true, true);
	}

	// Reserve buffer for temporarily storing intensities
	int nMaxTmpIntensities = 0;
	for (int i = 0; i < 2; i++) {
		cv::Size s = img[0].size();
		int nMaxCur = max(s.width, s.height) * 2;//(overestimate)
		if (nMaxCur > nMaxTmpIntensities) {
			nMaxTmpIntensities = nMaxCur;
		}
	}
	float* tmpIntensities = new float[nMaxTmpIntensities];
	float* tmpCount = new float[nMaxTmpIntensities];
	float* avg = new float[nSections];

	// Reserve memory for hash table.
	// The hash table consists of two halves, with
	// each half containing the lists for one image.
	DList dlist[2];
	for (int i = 0; i < 2; i++) {
		if (!dlist[i].create(nMaxDTokens, nL)) {
			cout << "failed to create hash table, you need more memory" << endl;
			delete[] avg;
			delete[] tmpCount;
			delete[] tmpIntensities;
			return -1;
		}
	}

	// Build the edges of graph, i.e. pairings of nodes,
	// (here simply a total, irreflexive relation),
	// Determine their respective d-token as hash-keys
	// and insert each pairing into the hash-table according to its d-token.
	// In this implementation we have a datastructure DList for each half of
	// the overall hash-table,
	// Each half holding the inverted lists of pairings for one image
	// (per bucket)
	for (int i = 0; i < 2; i++) { //for both images
		int nv = v[i].size(); //number of nodes
		//total irreflexive relation
		uint64 nPairings = ((nv - 1)*nv);
		printf("\n");
		if (nPairings < 1000000) {
			printf("number of strips to extract for image %d: %lld\n", i, nPairings);
		}
		else {
			printf("number of strips to extract for image %d: %.1f MegaStrips",
				i, (double)nPairings / (double)1000000.0);
		}
		uint64 count = 0;
		cout << endl;
		for (int j = 0; j < nv; j++) {
			for (int k = 0; k < nv; k++) {
				if (k != j) {
					// Compute and print progress...
					if (count % 10000 == 0 || count == nPairings - 1) {
						double percentage = (double)count / (double)nPairings;
						cout << '\r' << fixed << showpoint
							<< setprecision(2) << percentage * 100.0
							<< " % of d-tokens of image "
							<< i << " extracted";
					}
					count++;
					//extract d-token...
					uint64 dtoken;
					if (!extract_dtoken(dtoken, pyr[i], v[i].at(j), v[i].at(k),
						tmpIntensities, tmpCount, avg)) {
						cout << "failed to extract d-token\n";
						continue;
					}
					dlist[i].tryInsert(dtoken, j, k);
				}
			}
		}
	}

	Grid g(v[0].size(), v[1].size());
	if (!g.vote(dlist[0], dlist[1])) {
		cout << "voting failed\n";
	}
	g.visualize();
	//extract correspondence hypotheses...
	vector<pair<unsigned int, unsigned int> > correspondence_hypotheses;
	vector<float> quality;
	std::vector<cv::DMatch > grid_matches;
	std::vector<cv::DMatch > final_matches;
	if (!g.extractCorrespondenceHypotheses(grid_matches, qualityMode, nExtractOnlyNBest,
		bDiscardMultipleHits)) {
		cout << "failed ot extract correspondence hypotheses\n";
	}
	printf("\n");
	printf("preparing visualization...\n");
	//apply a geometric verification step for a better overview of the results, if desired
	if (gTransformationModel != NONE) {
		RansacKernel*pKernel = NULL;
		switch (gTransformationModel) {
		case AFFINE:
			pKernel = new RansacKernel_AT2(g_geometricVerification_threshDist);
			printf("geometric verification with AFFINE model\n");
			break;
		case HOMOGRAPHY:
			pKernel = new RansacKernel_Homography(g_geometricVerification_threshDist);
			printf("geometric verification with HOMOGRAPHY model\n");
			break;
		}

		Transformation2D* pTransformation2D = NULL;
		geometrically_verify_matches(pTransformation2D, final_matches,
			v, grid_matches, pKernel,
			g_nChunkSize, g_bChunkedCreation);
		if (pTransformation2D) {
			delete pTransformation2D;
		}
		if (pKernel) {
			delete pKernel;
			pKernel = NULL;
		}
	}
	else {
		printf("no geometric verification\n");
		final_matches = grid_matches;
	}

	cout << endl;
	cout << "Matches extracted\n";

	//-- Show detected matches
	cv::Mat img_8U[2];
	for (int i = 0; i < 2; i++) {
		if (!ensureGrayImageWithDepth(img_8U[i], CV_8U, img[i], NULL, false)) {
			// For some reason the original float-images would appear black
			// after the later call to drawMatches.
			cout << "failed to convert image for point detection" << endl;
		}
	}

	cv::Mat img_matches;
	switch (gVisualizationType) {
	case LINES:  drawMatches(img_8U[0], v[0], img_8U[1], v[1], final_matches, img_matches); break;
	case MESHES: drawMatchesThroughMeshes(img_8U[0], v[0], img_8U[1], v[1], final_matches,
		img_matches, g_MatchColor, g_singlePointColor1, g_singlePointColor2); break;
	}
	//这部分是把匹配的交点对输出到文本文档中
	string fname = "D:\\1ABpoint.txt";	//文本文档路径
	std::ofstream pout(fname, ios_base::trunc);
	for (int i = 0; i < final_matches.size(); i++) {

		const cv::KeyPoint&kp1 = v[0][final_matches[i].queryIdx];//v[0]指第一幅图片，[final_matches[i]是最终匹配的点列表，queryIdx是第一幅图片的点号
		const cv::KeyPoint&kp2 = v[1][final_matches[i].trainIdx]; //trainIdx是第二幅图片的点号
		std::cout << kp1.pt.x << " " << kp1.pt.y << " " << kp2.pt.x << " " << kp2.pt.y << endl;
		//pout时我为了方便找点也输出了两幅图的点号，但是到matlab里，是不要点号的，可以删掉两个点号
		pout<< kp1.pt.x << " " << kp1.pt.y << " "<< kp2.pt.x << " " << kp2.pt.y << endl;

	}
	cout << "FINISH" << endl;
	cout << final_matches.size() << endl;
	/************************************************/
	//这部分注释掉的是用现有的匹配特征点产生交点，因为特征点正确匹配，则交点肯定也是正确匹配的
	/*
	cout << "Start augment." << endl;
	vector<CvPoint2D32f> tempt;
	vector<CvPoint2D32f> tempt2;
	for (int i = 0; i < final_matches.size(); i = i + 4) {
		int num=(final_matches.size() / 4)*3;

		const cv::KeyPoint&kp_1 = v[0][final_matches[i].queryIdx];
		const cv::KeyPoint&kp_2 = v[0][final_matches[i + 1].queryIdx];
		const cv::KeyPoint&kp_3 = v[0][final_matches[i + 2].queryIdx];
		const cv::KeyPoint&kp_4 = v[0][final_matches[i + 3].queryIdx];
		tempt.push_back(CrossPoint1(kp_1, kp_2, kp_3, kp_4));
		tempt.push_back(CrossPoint1(kp_1, kp_3, kp_2, kp_4));
		tempt.push_back(CrossPoint1(kp_1, kp_4, kp_2, kp_3));
		const cv::KeyPoint&kp2_1 = v[1][final_matches[i].trainIdx];
		const cv::KeyPoint&kp2_2 = v[1][final_matches[i + 1].trainIdx];
		const cv::KeyPoint&kp2_3 = v[1][final_matches[i + 2].trainIdx];
		const cv::KeyPoint&kp2_4 = v[1][final_matches[i + 3].trainIdx];
		tempt2.push_back(CrossPoint1(kp2_1, kp2_2, kp2_3, kp2_4));
		tempt2.push_back(CrossPoint1(kp2_1, kp2_3, kp2_2, kp2_4));
		tempt2.push_back(CrossPoint1(kp2_1, kp2_4, kp2_2, kp2_3));
	}
	cout << "pic1" << endl;
	for (int i = 0; i < tempt.size(); i++) {
		cout << tempt[i].x <<"  "<< tempt[i].y <<" "<< tempt2[i].x << " " << tempt2[i].y<<endl;
		pout << tempt[i].x << " " << tempt[i].y <<" "<<tempt2[i].x << " " << tempt2[i].y << endl;
	}
/*	cout << "pic2" << endl;
	for (int i = 0; i < tempt2.size(); i++) {
		cout << tempt2[i].x <<" "<< tempt2[i].y << endl;
		pout << tempt2[i].x << " " << tempt2[i].y << endl;
	}

	/************************************************************/

	pout.close();
	imshow("Matches", img_matches);
	/*********************************************/
	//	这部分注释掉的是用现有点产生交点的可视化
	/*
	cv::Mat dstImage;
	cv::Mat dstImage2;
	dstImage = img_8U[0];
	dstImage2 = img_8U[1];
	for (size_t i = 0; i < tempt.size(); i++)
	{
		circle(dstImage, tempt[i], 3, CV_RGB(255, 255, 255), 2);
	}
	imshow("Matches_ADDED", dstImage);
	for (size_t i = 0; i < tempt2.size(); i++)
	{
		circle(dstImage2, tempt2[i], 3, CV_RGB(255, 255, 255), 2);
	}
	imshow("Matches_ADDED2", dstImage2);*/
	/************************************************/
	end = clock();
	dur = (double)(end - start);
	cout << "We have spend:" << (dur / CLOCKS_PER_SEC) << "  seconds" << endl;
	if (!outputFilename_VisualizedMatches.empty()) {
		cv::imwrite(outputFilename_VisualizedMatches.c_str(), img_matches);
		cout << endl;
		cout << "Visualization written to: "
			<< outputFilename_VisualizedMatches << endl;
	}
	cout << endl;
	if (bWait) {
		cout << endl
			<< "Press ESC to quit (while focus is in the top-level window.)\n";
		for (;;) {
			int c = cv::waitKey(0);
			if ((char)c == 27) { break; }
		}
	}

	// Clean up...
	delete[] avg;
	delete[] tmpCount;
	delete[] tmpIntensities;
	return 0;
}
