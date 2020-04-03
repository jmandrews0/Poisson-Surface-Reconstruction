// 211BProject.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <igl/cotmatrix.h>
#include <igl/octree.h>
#include <igl/knn.h>
#include <igl/dot.h>
#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/copyleft/marching_cubes.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stack>


double smoothF(const Eigen::Vector3d& q, const Eigen::Vector3d& p, double W)
{
	Eigen::Matrix3d Cov;
	Cov <<	1/1.0, 0, 0,
			0, 1/1.0, 0,
			0, 0, 1/1.0;
	double E = 2.71828;
	double PI = 3.14159;
	Eigen::Vector3d diff = (q - p);
	//std::cout << "det: " << Cov.determinant() << std::endl;
	return std::pow(E, -0.5*diff.transpose()*Cov*diff) / (std::sqrt(2 * PI*Cov.determinant())); // std::pow(W,3);
}

Eigen::Vector3d interpolate(const Eigen::Vector3d& q, const Eigen::MatrixXd& N, const Eigen::MatrixXd& CN, const Eigen::VectorXi& I)
{
	Eigen::Vector3d interpVector = Eigen::Vector3d(0, 0, 0);
	int pcount = 0;
	for (int s = 0; s < I.size(); s++)
	{
		int index = I(s);
		interpVector += N.row(index) * smoothF(q, CN.row(index), 1);
		pcount++;
	}
	return interpVector; //= pcount;
}


// second derivative of gaussian smoothF function
double laplacian(const Eigen::Vector3d& q, const Eigen::Vector3d& p, double W)
{
	//Eigen::Vector3d sigma(1, 1, 1);
	double E = 2.71828;
	double PI = 3.14159;
	double var = 1.0;
	Eigen::Vector3d diff = (q - p)/(2*W); //* 8.0;
	return (diff.transpose()*diff - std::pow(var, 2)) / std::pow(var, 4) * std::pow(E, -0.5 / std::pow(var, 2)*diff.transpose()*diff); // std::pow(W, 3);
}

// first derivative of gaussian smoothF function
double deriv(double q, double p, double W)
{
	double E = 2.71828;
	double PI = 3.14159;
	double var = 1.0;
	double diff = (q - p) / W;
	return (diff / std::pow(var, 2)) * std::pow(E, -0.5*std::pow(diff / var, 2)); // std::pow(W, 3);
}


int findCell(const Eigen::MatrixXi& CH, const Eigen::MatrixXd& CN, const Eigen::Vector3d& C)
{
	bool found = false;
	int cell = 0;
	while (!found)
	{
		//std::cout << "        cell = " << cell << std::endl;
		// is this node a child?
		bool leafN = true;
		for (int j = 0; j < 8; j++)
		{
			if (CH(cell, j) != -1) leafN = false;
		}
		if (leafN)
		{
			return cell;
		}
		//iterate deeper
		//std::cout << "        iterating deeper..." << std::endl;
		int index = 0;
		if (C(0) >= CN(cell, 0)) index += 1;
		if (C(1) >= CN(cell, 1)) index += 2;
		if (C(2) >= CN(cell, 2)) index += 4;
		cell = CH(cell, index);
	}
	return cell;
}


// get array of leaf node indices of octree
//leafNodes(O_CH, O_CN, O_L, L_CN);
void leafNodes(const Eigen::MatrixXi& CH, const Eigen::MatrixXd& CN, Eigen::VectorXi& Leaves, Eigen::MatrixXd& Leaves_CN)
{
	std::vector<int> L;
	std::vector<double> L_CN;
	for (int i = 0; i < CH.rows(); i++)
	{
		bool leaf = true;
		for (int j = 0; j < 8; j++)
		{
			if (CH(i,j) != -1) leaf = false;
		}
		if (leaf)
		{
			L.push_back(i);
			L_CN.push_back(CN(i,0));
			L_CN.push_back(CN(i,1));
			L_CN.push_back(CN(i,2));
			// get k nearest "same depth" nodes

		}
	}
	Leaves = Eigen::Map<Eigen::VectorXi>(L.data(), L.size());
	Leaves_CN = Eigen::Map<Eigen::MatrixXd>(L_CN.data(), 3, L.size()).transpose();
}


void getNeighborCells(const Eigen::VectorXi& L, Eigen::MatrixXi& CH, const Eigen::MatrixXd& CN, const Eigen::VectorXd& W, Eigen::MatrixXi& Neighbors)
{
	Neighbors = Eigen::MatrixXi(L.rows(), 7);
	std::map<int, int> leaf;
	for (int i = 0; i < L.size(); i++)
	{
		leaf[L(i)] = i;
	}

	for (int i = 0; i < L.size(); i++)
	{
		//std::cout << "i = " << i << " -----------------------------------" << std::endl;
		// find nearest neighbors for this cell
		Neighbors(i, 0) = i;
		int index = L(i);
		Eigen::Vector3d C = CN.row(index);
		std::vector<Eigen::Vector3d> searchCenters;
		Eigen::Vector3d X(W(index), 0, 0);
		Eigen::Vector3d Y(0, W(index), 0);
		Eigen::Vector3d Z(0, 0, W(index));
		searchCenters.push_back(C - X);
		searchCenters.push_back(C + X);
		searchCenters.push_back(C - Y);
		searchCenters.push_back(C + Y);
		searchCenters.push_back(C - Z);
		searchCenters.push_back(C + Z);
		// iterate through all search centers looking for match
		for (int j=0; j < searchCenters.size(); j++)
		{
			int cell = findCell(CH, CN, searchCenters.at(j));
			Neighbors(i, j + 1) = leaf.at(cell);
		}
	}
}

int level(double width, double total)
{
	return (int)std::log2(total / width);
}

Eigen::Vector3d gradientField(	const Eigen::Vector3d& q, // point to interpolate
								const Eigen::MatrixXd& V, // all vertices
								const Eigen::MatrixXd& N, // all normals
								const std::vector<std::vector<int>>& PI, // indices of points in each octree cell
								const Eigen::MatrixXi& CH, // children for each octree cell
								const Eigen::MatrixXd& CN, // center of each octree cell
								const Eigen::VectorXd& W, // nearest neighbors of each octree cell
								const Eigen::VectorXi& L) // leaf node list
{
	int pcount = 1;
	Eigen::Vector3d interpVector(0,0,0);
	/*
	for (int c = 0; c < nearest.size(); c++)
	{
		int index = nearest(c);
		std::vector<int> points = PI.at(index);
		for (int s = 0; s < points.size(); s++)
		{
			interpVector += N.row(points.at(s)) * smoothF(q, CN.row(index), W(index)) * W(index);
			pcount++;
		}
	}
	*/
	// get max depth
	int max = 0;
	for (int i = 0; i < L.size(); i++)
	{
		int lev = level(W(L(i)), W(0));
		if (lev > max) max = lev;
	}

	int index = findCell(CH, CN, q);
	int lev = level(W(index), W(0));
	if (lev >= max-1)
	{
		for (int s = 0; s < V.rows(); s++)
		{
			int index = findCell(CH, CN, V.row(s)); // get index of octree cell for sample s
			interpVector += N.row(s) * smoothF(q, CN.row(index), 1.0); //* W(index);
			pcount++;
		}
	}

	return interpVector / pcount;
}

void generateL(int x_res, int y_res, int z_res, Eigen::SparseMatrix<double>& Lap)
{
	int size = x_res * y_res * z_res;
	Lap = Eigen::SparseMatrix<double>(size, size);
	std::vector<Eigen::Triplet<double>> coeff;

	for (int r = 0; r < size; r++)
	{
		if (r%x_res == 0) {
			coeff.push_back(Eigen::Triplet<double>(r,r,1e9));//Lap.insert(r, r) = 1e9;
			continue;
		}
		if (r%x_res == x_res-1) {
			coeff.push_back(Eigen::Triplet<double>(r, r, 1e9));//Lap.insert(r, r) = 1e9;
			continue;
		}
		if ((r / x_res) % y_res == 0) {
			coeff.push_back(Eigen::Triplet<double>(r, r, 1e9));//Lap.insert(r, r) = 1e9;
			continue;
		}
		if ((r / x_res) % y_res == y_res-1) {
			coeff.push_back(Eigen::Triplet<double>(r, r, 1e9));//Lap.insert(r, r) = 1e9;
			continue;
		}
		if ((r / (x_res*y_res)) % z_res == 0) {
			coeff.push_back(Eigen::Triplet<double>(r, r, 1e9));//Lap.insert(r, r) = 1e9;
			continue;
		}
		if ((r / (x_res*y_res)) % z_res == z_res-1) {
			coeff.push_back(Eigen::Triplet<double>(r, r, 1e9));//Lap.insert(r, r) = 1e9;
			continue;
		}
		coeff.push_back(Eigen::Triplet<double>(r, r, 6));//Lap.insert(r, r) = 6;
		// x neighbors
		if(r < size-1) coeff.push_back(Eigen::Triplet<double>(r, r+1, -1));
		if(r > 0) coeff.push_back(Eigen::Triplet<double>(r, r-1, -1));
		// y neighbors
		if(r < size-x_res) coeff.push_back(Eigen::Triplet<double>(r, r+x_res, -1));
		if(r > x_res) coeff.push_back(Eigen::Triplet<double>(r, r-x_res, -1));
		// z neighbors
		if(r < size- x_res * y_res) coeff.push_back(Eigen::Triplet<double>(r, r+x_res*y_res, -1));
		if(r > x_res * y_res) coeff.push_back(Eigen::Triplet<double>(r, r-x_res*y_res, -1));
		/*
		// diagonal y neighbors
		if (r < size - (x_res-1)) coeff.push_back(Eigen::Triplet<double>(r, r + (x_res-1), -1));
		if (r < size - (x_res+1)) coeff.push_back(Eigen::Triplet<double>(r, r + (x_res+1), -1));
		if (r > (x_res-1)) coeff.push_back(Eigen::Triplet<double>(r, r - (x_res-1), -1));
		if (r > (x_res+1)) coeff.push_back(Eigen::Triplet<double>(r, r - (x_res+1), -1));
		// diagonal z neighbors
		if (r < size - (x_res*y_res-1)) coeff.push_back(Eigen::Triplet<double>(r, r + (x_res*y_res-1), -1));
		if (r < size - (x_res*y_res+1)) coeff.push_back(Eigen::Triplet<double>(r, r + (x_res*y_res+1), -1));
		if (r > (x_res*y_res-1)) coeff.push_back(Eigen::Triplet<double>(r, r - (x_res*y_res-1), -1));
		if (r > (x_res*y_res+1)) coeff.push_back(Eigen::Triplet<double>(r, r - (x_res*y_res+1), -1));
		// diagonal x neighbors
		if (r < size - (x_res * y_res - x_res)) coeff.push_back(Eigen::Triplet<double>(r, r + (x_res * y_res - x_res), -1));
		if (r < size - (x_res * y_res + x_res)) coeff.push_back(Eigen::Triplet<double>(r, r + (x_res * y_res + x_res), -1));
		if (r > (x_res * y_res - x_res)) coeff.push_back(Eigen::Triplet<double>(r, r - (x_res * y_res - x_res), -1));
		if (r > (x_res * y_res + x_res)) coeff.push_back(Eigen::Triplet<double>(r, r - (x_res * y_res + x_res), -1));
		// x y z diagonals
		if (r < size - (x_res - x_res*y_res - 1)) coeff.push_back(Eigen::Triplet<double>(r, r + (x_res - x_res * y_res - 1), -1));
		if (r < size - (x_res - x_res*y_res + 1)) coeff.push_back(Eigen::Triplet<double>(r, r + (x_res - x_res * y_res + 1), -1));
		if (r > (x_res - x_res * y_res - 1)) coeff.push_back(Eigen::Triplet<double>(r, r - (x_res - x_res * y_res - 1), -1));
		if (r > (x_res - x_res * y_res + 1)) coeff.push_back(Eigen::Triplet<double>(r, r - (x_res - x_res * y_res + 1), -1));
		if (r < size - (x_res + x_res * y_res - 1)) coeff.push_back(Eigen::Triplet<double>(r, r + (x_res + x_res * y_res - 1), -1));
		if (r < size - (x_res + x_res * y_res + 1)) coeff.push_back(Eigen::Triplet<double>(r, r + (x_res + x_res * y_res + 1), -1));
		if (r > (x_res + x_res * y_res - 1)) coeff.push_back(Eigen::Triplet<double>(r, r - (x_res + x_res * y_res - 1), -1));
		if (r > (x_res + x_res * y_res + 1)) coeff.push_back(Eigen::Triplet<double>(r, r - (x_res + x_res * y_res + 1), -1));
		*/

	}
	Lap.setFromTriplets(coeff.begin(), coeff.end());
}


double minWidth(const Eigen::VectorXd& W)
{
	double min = W(0);
	for (int i = 1; i < W.rows(); i++)
	{
		if (W(i) <= min) min = W(i);
	}
	return min;
}


void makeDenseField(	const Eigen::MatrixXd& V,
						const Eigen::MatrixXd& N,
						const Eigen::VectorXd& W,
						const Eigen::VectorXi& L,
						const Eigen::MatrixXi& I,
						Eigen::VectorXd& denseSol,
						Eigen::MatrixXd& denseCenters,
						Eigen::MatrixXd& denseVectors,
						std::map<int,int>& sampleToDense,
						int& x_res, int& y_res, int& z_res)
{
	// get max depth
	int max = 0;
	for (int i = 0; i < L.size(); i++)
	{
		int lev = level(W(L(i)), W(0));
		if (lev > max) max = lev;
	}
	std::cout << "max depth: " << max << std::endl;
	if (max > 6) max = 6;
	//max = max - 1;

	x_res = std::pow(2, max);
	y_res = std::pow(2, max);
	z_res = std::pow(2, max);
	denseSol = Eigen::VectorXd(x_res*y_res*z_res);
	denseCenters = Eigen::MatrixXd(x_res*y_res*z_res, 3);
	denseVectors = Eigen::MatrixXd(x_res*y_res*z_res, 3);
	double step = W(0) / (double)std::pow(2, max);
	Eigen::Vector3d X(1, 0, 0);
	Eigen::Vector3d Y(0, 1, 0);
	Eigen::Vector3d Z(0, 0, 1);
	Eigen::Vector3d start = X * step / 2.0 + Y * step / 2.0 + Z * step / 2.0 - X * W(0) / 2.0 - Y * W(0) / 2.0 - Z * W(0) / 2.0;

	// initialize values in dense field
	int count = 0;
	for (int x = 0; x < x_res; x++)
	{
		for (int y = 0; y < y_res; y++)
		{
			for (int z = 0; z < z_res; z++)
			{
				Eigen::Vector3d C = x * X*step + y * Y*step + z * Z*step + start;
				denseSol(count) = 0;
				denseCenters.row(count) = C;
				denseVectors.row(count) = Eigen::Vector3d(0, 0, 0);
				count++;
			}
		}
	}
	// fill in cells occupied by sample points
	int totalSize = x_res * y_res * z_res;
	std::cout << "NUMBER OF SAMPLES: " << V.rows() << std::endl;
	for (int s = 0; s < V.rows(); s++)
	{
		//std::cout << "sample " << s << std::endl;
		Eigen::Vector3d sample = V.row(s).transpose() + X * W(0) / 2.0 + Y * W(0) / 2.0 + Z * W(0) / 2.0;
		int index = z_res * y_res * ((int)(sample(0) / step)) + z_res * ((int)(sample(1) / step)) + (int)(sample(2) / step);
		Eigen::Vector3d center = start + (index%z_res)*Z*step + ((int)(index / z_res)%y_res)*Y*step + ((int)(index / (z_res*y_res))%x_res)*X*step;
		sampleToDense[s] = index;

		denseVectors.row(index) = interpolate(center, N, V, I.row(s));
		if (index > 0) {
			int ind = index - 1;
			center = start + (ind%z_res)*Z*step + ((int)(ind / z_res) % y_res)*Y*step + ((int)(ind / (z_res*y_res)) % x_res)*X*step;
			denseVectors.row(ind) = interpolate(center, N, V, I.row(s));
		}
		if (index < totalSize - 1) {
			int ind = index + 1;
			center = start + (ind%z_res)*Z*step + ((int)(ind / z_res) % y_res)*Y*step + ((int)(ind / (z_res*y_res)) % x_res)*X*step;
			denseVectors.row(ind) = interpolate(center, N, V, I.row(s));
		}
		if (index > z_res){
			int ind = index - z_res;
			center = start + (ind%z_res)*Z*step + ((int)(ind / z_res) % y_res)*Y*step + ((int)(ind / (z_res*y_res)) % x_res)*X*step;
			denseVectors.row(ind) = interpolate(center, N, V, I.row(s));
		}
		if (index < totalSize-z_res) {
			int ind = index + z_res;
			center = start + (ind%z_res)*Z*step + ((int)(ind / z_res) % y_res)*Y*step + ((int)(ind / (z_res*y_res)) % x_res)*X*step;
			denseVectors.row(ind) = interpolate(center, N, V, I.row(s));
		}
		if (index > z_res*y_res) {
			int ind = index - z_res * y_res;
			center = start + (ind%z_res)*Z*step + ((int)(ind / z_res) % y_res)*Y*step + ((int)(ind / (z_res*y_res)) % x_res)*X*step;
			denseVectors.row(ind) = interpolate(center, N, V, I.row(s));
		}
		if (index < totalSize - z_res*y_res) {
			int ind = index + z_res * y_res;
			center = start + (ind%z_res)*Z*step + ((int)(ind / z_res) % y_res)*Y*step + ((int)(ind / (z_res*y_res)) % x_res)*X*step;
			denseVectors.row(ind) = interpolate(center, N, V, I.row(s));
		}
	}
	for (int s = 0; s < denseVectors.rows(); s++)
	{
		denseSol(s) = denseVectors.row(s).norm();
	}
}


double calcIsovalue(const Eigen::MatrixXd& V, const Eigen::MatrixXi& CH, const Eigen::MatrixXd& CN, const Eigen::VectorXi& L, const Eigen::VectorXd& sol, const std::map<int, int>& sampleToDense)
{
	std::map<int, int> leaf;
	for (int i = 0; i < L.size(); i++)
	{
		leaf[L(i)] = i;
	}

	int cell = 0;
	double sum = 0;
	int count = 0;
	for (int i = 0; i < V.rows(); i++)
	{
		//cell = findCell(CH, CN, V.row(i));
		//std::cout << "found cell " << cell << std::endl;
		//int l = leaf.at(cell);
		//std::cout << "got leaf for cell " << l << std::endl;
		if (sampleToDense.find(i) != sampleToDense.end()) {
			//std::cout << "    dense has leaf" << i << std::endl;
			int index = sampleToDense.at(i);
			//std::cout << "    got index for leaf" << index << std::endl;
			sum += sol(index);
			count++;
		}
		//Eigen::Vector3d loc = CN.row(cell);
		// search for correct solution index
		//std::cout << "added to sum" << std::endl;
	}
	return sum / count;
}


void applyEdgeConstraints(Eigen::MatrixXd& MatL, const Eigen::VectorXi& L, const Eigen::MatrixXi& CH, const Eigen::MatrixXd& CN, Eigen::MatrixXd& Edges)
{
	std::vector<double> edges;
	std::map<int, int> leaf;
	for (int i = 0; i < L.size(); i++)
	{
		leaf[L(i)] = i;
	}
	// search for all 8 corner nodes

	bool leafN = false;
	int cell = 0;
	while (!leafN)
	{
		for (int j = 0; j < 8; j++)
		{
			if (CH(cell, j) == -1) leafN = true;
		}
		if (!leafN) cell = CH(cell, 0);
	}
	//std::cout << "selected cell: " << CN.row(cell) << std::endl;

	for (int j = 0; j < L.rows(); j++)
	{
		int l = L(j);
		if (std::abs(CN(l, 0)) >= std::abs(CN(cell,0))) { MatL(j, j) = 1e9; 
			edges.push_back(CN(l, 0)); edges.push_back(CN(l, 1)); edges.push_back(CN(l, 2));
		}
		else if (std::abs(CN(l, 1)) >= std::abs(CN(cell, 1))) { MatL(j, j) = 1e9;
			edges.push_back(CN(l, 0)); edges.push_back(CN(l, 1)); edges.push_back(CN(l, 2));
		}
		else if (std::abs(CN(l, 2)) >= std::abs(CN(cell, 2))) { MatL(j, j) = 1e9;
			edges.push_back(CN(l, 0)); edges.push_back(CN(l, 1)); edges.push_back(CN(l, 2));
		}
	}
		/*
		if (i == 0)
		{
			for (int j = 0; j < CN.rows(); j++)
			{
				int l = leaf.at(cell);
				if (CN.row(j)[0] < CN.row(cell)[0]) { MatL(l, l) = 1e9; count++; 
					edges.push_back(CN(j,0)); edges.push_back(CN(j, 1)); edges.push_back(CN(j, 2));
				}
				else if (CN.row(j)[1] < CN.row(cell)[1]) { MatL(l, l) = 1e9; count++; 
					edges.push_back(CN(j, 0)); edges.push_back(CN(j, 1)); edges.push_back(CN(j, 2));
				}
				else if (CN.row(j)[2] < CN.row(cell)[2]) { MatL(l, l) = 1e9; count++;
					edges.push_back(CN(j, 0)); edges.push_back(CN(j, 1)); edges.push_back(CN(j, 2));
				}
			}
		}
		else if (i == 7)
		{
			for (int j = 0; j < CN.rows(); j++)
			{
				int l = leaf.at(cell);
				if (CN.row(j)[0] > CN.row(cell)[0]) { MatL(l, l) = 1e9; count++;
					edges.push_back(CN(j, 0)); edges.push_back(CN(j, 1)); edges.push_back(CN(j, 2));
				}
				else if (CN.row(j)[1] > CN.row(cell)[1]) { MatL(l, l) = 1e9; count++;
					edges.push_back(CN(j, 0)); edges.push_back(CN(j, 1)); edges.push_back(CN(j, 2));
				}
				else if (CN.row(j)[2] > CN.row(cell)[2]) { MatL(l, l) = 1e9; count++;
					edges.push_back(CN(j, 0)); edges.push_back(CN(j, 1)); edges.push_back(CN(j, 2));
				}
			}
		}
		*/
	Edges = Eigen::Map<Eigen::MatrixXd>(edges.data(), 3, edges.size()/3).transpose();
}


void extraPoints(const Eigen::MatrixXd& V, Eigen::MatrixXd& Extra)
{
	Extra = Eigen::MatrixXd(V.rows()+6, 3);
	double minX = V(0, 0); 	double maxX = V(0, 0);
	double minY = V(0, 1);	double maxY = V(0, 1);
	double minZ = V(0, 2);	double maxZ = V(0, 2);
	for (int i = 0; i < V.rows(); i++)
	{
		Extra.row(i) = V.row(i);
		if (V(i, 0) > maxX) maxX = V(i, 0);
		if (V(i, 1) > maxY) maxY = V(i, 1);
		if (V(i, 2) > maxZ) maxZ = V(i, 2);
		if (V(i, 0) < minX) minX = V(i, 0);
		if (V(i, 1) < minY) minY = V(i, 1);
		if (V(i, 2) < minZ) minZ = V(i, 2);
	}
	double diffX = maxX - minX;
	double diffY = maxY - minY;
	double diffZ = maxZ - minZ; 
	Extra.row(V.rows()) = Eigen::Vector3d(maxX + diffX / 8, 0, 0);
	Extra.row(V.rows()+1) = Eigen::Vector3d(minX - diffX / 8, 0, 0);
	Extra.row(V.rows()+2) = Eigen::Vector3d(0, maxY + diffY / 8, 0);
	Extra.row(V.rows()+3) = Eigen::Vector3d(0, minY - diffY / 8, 0);
	Extra.row(V.rows()+4) = Eigen::Vector3d(0, 0, maxZ + diffZ / 8);
	Extra.row(V.rows()+5) = Eigen::Vector3d(0, 0, minZ - diffZ / 8);
}
//------------------------------------------------------------------------------------------------------

int main()
{
	// load sample points
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	igl::readOBJ("hourglass.obj",V,F);

	// Build octree for k nearest neighbors S_I
	std::vector<std::vector<int > > O_PI;
	Eigen::MatrixXi O_CH;
	Eigen::MatrixXd O_CN;
	Eigen::VectorXd O_W;
	igl::octree(V, O_PI, O_CH, O_CN, O_W);
	// get list of leaf nodes only
	Eigen::VectorXi O_L;
	Eigen::MatrixXd O_LCN;
	leafNodes(O_CH, O_CN, O_L, O_LCN);
	// find k nearest neighbors for sample points
	Eigen::MatrixXi S_I;
	int k = 8;
	igl::knn(V, k, O_PI, O_CH, O_CN, O_W, S_I);
	std::cout << "found k nearest neighbors" << std::endl;
	
	// Build octree for sample points
	std::vector<std::vector<int > > S_PI;
	Eigen::MatrixXi S_CH;
	Eigen::MatrixXd S_CN;
	Eigen::VectorXd S_W;
	Eigen::MatrixXd Extra;
	extraPoints(V, Extra);
	igl::octree(Extra, S_PI, S_CH, S_CN, S_W);
	std::cout << "second octree" << std::endl;
	// get list of leaf nodes only
	Eigen::VectorXi S_L;
	Eigen::MatrixXd S_LCN;
	leafNodes(S_CH, S_CN, S_L, S_LCN);
	
	// CALCULATE NORMALS FOR EACH POINT
	Eigen::MatrixXd N(V.rows(), 3); // normals to be calculated
	Eigen::MatrixXd M(k, 3); // covariance matrix to be calculated
	
	for (int r = 0; r < V.rows(); r++)
	{
		// calculate mean of points
		Eigen::Vector3d mean(0, 0, 0);
		for (int v = 0; v < S_I.cols(); v++)
		{
			mean += V.row(S_I(r, v));
		}
		mean /= S_I.cols();
		
		// calculate matrix M
		Eigen::Vector3d diff(0, 0, 0);
		for (int v = 0; v < S_I.cols(); v++)
		{
			diff = V.row(S_I(r, v)) - mean.transpose();
			M.row(v) = diff;
		}

		// SVD decomposition
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::MatrixXd V = svd.matrixV();
		Eigen::VectorXd rhs = Eigen::VectorXd::Zero(k); rhs(0) = 1; // will have to be more dimensions for larger values of k
		N.row(r) = svd.solve(rhs).normalized();
	}
	std::cout << "calculated normals" << std::endl;

	// GENERATE VECTOR FIELD WITH CONSTANT RESOLUTION
	Eigen::VectorXd denseField;
	Eigen::MatrixXd denseCenters;
	Eigen::MatrixXd denseVectors;
	std::map<int, int> sampleToDense;
	int x_res; int y_res; int z_res;
	makeDenseField(V, N, S_W, S_L, S_I, denseField, denseCenters, denseVectors, sampleToDense, x_res, y_res, z_res);
	std::cout << "denseSol: " << denseField.size() << "denseCenters" << denseCenters.size() << std::endl;
	std::cout << x_res << ", " << y_res << ", " << z_res << std::endl;
	
	// CREATE LAPLACIAN MATRIX
	Eigen::SparseMatrix<double> MatL; // make SparseMatrix<double>
	generateL(x_res, y_res, z_res, MatL);
	std::cout << "Generated Laplacian" << std::endl;

	// SOLVE min||Lx-v||^2 USING SVD
	Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> lscg;
	lscg.compute(MatL);
	Eigen::VectorXd sol = lscg.solve(denseField);
	std::cout << "Computed Indicator Function" << std::endl;

	// APPROXIMATE ISOVALUE NEAR SURFACE
	double isovalue = calcIsovalue(V, S_CH, S_CN, S_L, sol, sampleToDense);
	std::cout << "isovalue = " << isovalue << std::endl;

	// MARCHING CUBES ALGORITHM
	Eigen::MatrixXd V_Recon;
	Eigen::MatrixXi F_Recon;
	igl::copyleft::marching_cubes(sol, denseCenters, x_res, y_res, z_res, isovalue, V_Recon, F_Recon);

	std::cout << "finished marching cubes" << std::endl;
	
	// PLOTTING DATA AND MESH
	const Eigen::RowVector3d green(0.2, 0.9, 0.2), blue(0.2, 0.2, 0.8), red(0.8, 0.2, 0.2), white(0.9, 0.9, 0.9), yellow(0.9,0.9,0.2);
	igl::opengl::glfw::Viewer viewer;
	/*
	for (int j = 0; j < S_L.size(); j++)
	{
		int i = S_L(j);
		Eigen::MatrixXd OctreeV1(8, 3);
		Eigen::MatrixXd OctreeV2(8, 3);
		Eigen::Vector3d X(S_W(i), 0.0, 0.0);
		Eigen::Vector3d Y(0.0, S_W(i), 0.0);
		Eigen::Vector3d Z(0.0, 0.0, S_W(i));
		OctreeV1 << (S_CN.row(i).transpose() - X / 2.0 - Y / 2.0 - Z / 2.0).transpose(),
			(S_CN.row(i).transpose() - X / 2.0 - Y / 2.0 + Z / 2.0).transpose(),
			(S_CN.row(i).transpose() - X / 2.0 + Y / 2.0 - Z / 2.0).transpose(),
			(S_CN.row(i).transpose() + X / 2.0 + Y / 2.0 + Z / 2.0).transpose(),
			(S_CN.row(i).transpose() + X / 2.0 - Y / 2.0 - Z / 2.0).transpose(),
			(S_CN.row(i).transpose() + X / 2.0 - Y / 2.0 + Z / 2.0).transpose(),
			(S_CN.row(i).transpose() + X / 2.0 + Y / 2.0 - Z / 2.0).transpose(),
			(S_CN.row(i).transpose() + X / 2.0 + Y / 2.0 + Z / 2.0).transpose();

		OctreeV2 << (S_CN.row(i).transpose() - X / 2.0 + Y / 2.0 - Z / 2.0).transpose(),
			(S_CN.row(i).transpose() - X / 2.0 + Y / 2.0 + Z / 2.0).transpose(),
			(S_CN.row(i).transpose() - X / 2.0 + Y / 2.0 + Z / 2.0).transpose(),
			(S_CN.row(i).transpose() + X / 2.0 + Y / 2.0 - Z / 2.0).transpose(),
			(S_CN.row(i).transpose() + X / 2.0 + Y / 2.0 - Z / 2.0).transpose(),
			(S_CN.row(i).transpose() + X / 2.0 + Y / 2.0 + Z / 2.0).transpose(),
			(S_CN.row(i).transpose() - X / 2.0 + Y / 2.0 - Z / 2.0).transpose(),
			(S_CN.row(i).transpose() - X / 2.0 + Y / 2.0 + Z / 2.0).transpose();
		viewer.data().add_edges(OctreeV1, OctreeV2, white);
	}
	*/
	Eigen::MatrixXd intensity(sol.rows(), 3);
	for (int i = 0; i < sol.rows(); i++)
	{
		intensity.row(i) = Eigen::Vector3d(0,sol(i),0);
	}
	Eigen::MatrixXd VField(sol.rows(), 3);
	for (int i = 0; i < sol.rows(); i++)
	{
		VField.row(i) = Eigen::Vector3d(0, denseField(i), 0);
	}
	
	std::ofstream myfile;
	myfile.open("sol.txt");
	myfile << sol;
	myfile.close();

	// Plot the mesh
	viewer.data().set_mesh(V_Recon, F_Recon);
	viewer.data().set_face_based(true);
	viewer.data().add_points(V, red);
	//viewer.data().add_edges(denseCenters + denseVectors, denseCenters, green);
	//viewer.data().add_edges(V + N, V, green);
	//std::cout << "solution space: "<< denseSol.rows() << ", centers: " << denseCenters.rows() << ", intensity: " << intensity.size() << std::endl;
	//viewer.data().add_edges(denseCenters + intensity, denseCenters, white);
	viewer.launch();

	return 0;
}
