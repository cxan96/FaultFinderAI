/**
 * 
 */
package domain.models;

import org.deeplearning4j.nn.graph.ComputationGraph;

/**
 * @author m.c.kunkel
 *
 */
public class CLASModelFactory {

	int height;
	int width;
	int numChannels;
	int numLabels;

	public CLASModelFactory() {

	}

	public ComputationGraph getModel(String type) {
		ComputationGraph computationGraph = null;
		/**
		 * This is the base case for just one superlayer
		 */
		if (type.isEmpty()) {
			computationGraph = ModelFactory.computationGraphModel(height, width, numChannels, numLabels);
		} else if (type.equalsIgnoreCase("clasdc")) {
			computationGraph = ModelFactory.computationGraphModel(height, width, numChannels, numLabels);
		} else if (type.equalsIgnoreCase("clasRegion")) {
			computationGraph = ModelFactory.computationGraphModel(height, width, numChannels, numLabels);
		} else if (type.equalsIgnoreCase("clas")) {
			computationGraph = ModelFactory.computationGraphModel(height, width, numChannels, numLabels);
		} else {
			throw new IllegalArgumentException("Invalid input: " + type);

		}

		return computationGraph;

	}

}
