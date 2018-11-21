/**
 * 
 */
package domain.models;

import org.deeplearning4j.nn.graph.ComputationGraph;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

/**
 * @author m.c.kunkel
 *
 */
@RequiredArgsConstructor
public class CLASModelFactory {

	private final int height;
	private final int width;
	private final int numChannels;

	@Getter
	private int gridWidth;
	@Getter
	private int gridHeight;

	public ComputationGraph getModel(String modelType) {
		ComputationGraph computationGraph = null;
		/**
		 * This is the base case for just one superlayer
		 */
		if (modelType.isEmpty()) {
			this.gridHeight = 3;
			this.gridWidth = 28;
			computationGraph = Models.KunkelPetersYolo(height, width, numChannels);
		} else if (modelType.equalsIgnoreCase("clasdc")) {
			this.gridHeight = 7;
			this.gridWidth = 45;
			// computationGraph = Models.DCModel(height, width, numChannels);
			computationGraph = Models.KunkelPetersUYolo(height, width, numChannels);
		} else if (modelType.equalsIgnoreCase("clasRegion")) {
			this.gridHeight = 36;
			this.gridWidth = 28;
			computationGraph = Models.RegionhModel(height, width, numChannels);
		} else if (modelType.equalsIgnoreCase("clas")) {
			this.gridHeight = 36 * 3;
			this.gridWidth = 28;
			computationGraph = Models.CLASModel(height, width, numChannels);
		} else {
			throw new IllegalArgumentException("Invalid input: " + modelType);

		}

		return computationGraph;

	}

}
