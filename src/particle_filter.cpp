/*
* particle_filter.cpp
*
*  Created on: Dec 12, 2016
*      Author: Tiffany Huang
*/

#define _USE_MATH_DEFINES
#include <cmath> 

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 110;

	default_random_engine gen;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);


	for (int i = 0; i < num_particles; i++) {

		Particle p;

		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);
	}


	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++) {

		if (fabs(yaw_rate) > 0.00001) {
			double const v_yaw = velocity / yaw_rate;
			double const theta_dt = particles[i].theta + yaw_rate * delta_t;
			particles[i].x += v_yaw * (sin(theta_dt) - sin(particles[i].theta)) + dist_x(gen);
			particles[i].y += v_yaw * (cos(particles[i].theta) - cos(theta_dt)) + dist_y(gen);
			particles[i].theta += (yaw_rate * delta_t) + dist_theta(gen);
		}
		else {
			double const v_dt = velocity * delta_t;
			particles[i].x += v_dt * cos(particles[i].theta) + dist_x(gen);
			particles[i].y += v_dt * sin(particles[i].theta) + dist_y(gen);
			particles[i].theta += dist_theta(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.


	for (int i = 0; i < observations.size(); i++) {

		LandmarkObs tobs = observations[i];
		double prev_dist = 99999;
		int ldm_id = 0;

		for (int j = 0; j < predicted.size(); j++) {
			LandmarkObs lobs = predicted[j];
			double cur_dist = dist(tobs.x, tobs.y, lobs.x, lobs.y);
			if (cur_dist < prev_dist) {
				prev_dist = cur_dist;
				ldm_id = lobs.id;
			}
		}
		observations[i].id = ldm_id;
	} // end obs

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


	/* Transformation/Rotation and association */
	for (int i = 0; i < num_particles; i++) {

		std::vector<LandmarkObs> predicted;

		double px = particles[i].x;
		double py = particles[i].y;
		double ptheta = particles[i].theta;

		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

			// Get Landmark coordinates
			float lmx = map_landmarks.landmark_list[j].x_f;
			float lmy = map_landmarks.landmark_list[j].y_f;
			int lm_id = map_landmarks.landmark_list[j].id_i;

			if (fabs(lmx - px) <= sensor_range && fabs(lmy - py) <= sensor_range) {
				predicted.push_back(LandmarkObs{ lm_id, lmx, lmy });
			}
		}
		// Get transformed observation
		vector<LandmarkObs> observation_t;
		for (unsigned int j = 0; j < observations.size(); j++) {
			double tobs_x = px + cos(ptheta)*observations[j].x - sin(ptheta)*observations[j].y;
			double tobs_y = py + sin(ptheta)*observations[j].x + cos(ptheta)*observations[j].y;
			observation_t.push_back(LandmarkObs{ observations[j].id, tobs_x, tobs_y });
		}

		//Get the association with nearest landmark
		dataAssociation(predicted, observation_t);

		//Calculate weight
		double total_wt = 1.0;
		for (int j = 0; j < observation_t.size(); j++) {
			double to_x, to_y, mu_x, mu_y;
			for (int k = 0; k < predicted.size(); k++) {
				if (predicted[k].id == observation_t[j].id) {
					mu_x = predicted[k].x;
					mu_y = predicted[k].y;
				}
			}
			to_x = observation_t[j].x;
			to_y = observation_t[j].y;
			double wt = (1 / (2 * M_PI*std_landmark[0] * std_landmark[1])) * exp(-(pow(mu_x - to_x, 2) / (2 * pow(std_landmark[0], 2)) + (pow(mu_y - to_y, 2) / (2 * pow(std_landmark[1], 2)))));
			if (wt > 0) {
				total_wt *= wt;
			}
		}
		//Final weight of the particle
		particles[i].weight = total_wt;
		weights.push_back(total_wt);

	}// end of num_particles
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> particles_resampled;
	default_random_engine gen;
	discrete_distribution<int> wt_dist(weights.begin(), weights.end());

	for (int i = 0; i < num_particles; i++) {
		particles_resampled.push_back(particles[wt_dist(gen)]);
	}

	particles.clear();
	particles = particles_resampled;
	weights.clear();
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
