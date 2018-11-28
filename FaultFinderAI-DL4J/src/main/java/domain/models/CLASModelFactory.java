/**
 * 
 */
package domain.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;

import clasDC.faults.FaultNames;
import clasDC.objects.CLASObject;
import clasDC.objects.DriftChamber;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import utils.FaultUtils;

/**
 * @author m.c.kunkel
 *
 */
public class CLASModelFactory {

	private final int height;
	private final int width;
	private final int numChannels;
	private int numClasses;
	private double[][] priors;
	private String modelType;

	@Getter
	private int gridWidth;
	@Getter
	private int gridHeight;
	@Getter
	ComputationGraph computationGraph = null;

	@Builder
	private CLASModelFactory(CLASObject clasOject) {
		this.height = clasOject.getHeight();
		this.width = clasOject.getWidth();
		this.numChannels = clasOject.getNchannels();
		this.numClasses = clasOject.getDesiredFaults().stream().distinct().collect(Collectors.toList()).size();
		this.priors = clasOject.getPriors();
		this.modelType = clasOject.getObjectType();
		setModel();
	}

	public void setModel() {
		ComputationGraph temp = null;
		/*
		 * This is the base case for just one superlayer
		 */
		if (modelType.equalsIgnoreCase("SuperLayer")) {
			temp = Models.KunkelPetersUYolo4SL(height, width, numChannels, numClasses, priors);
			setGridDimensions(temp);
			double[][] priorBoxes = FaultUtils.getPriors(priors,
					new double[][] { { height / this.gridHeight, width / this.gridWidth } });// FaultUtils.allPriors
			this.computationGraph = Models.KunkelPetersUYolo4SL(height, width, numChannels, numClasses, priorBoxes);
			System.out.println(computationGraph.summary(InputType.convolutional(height, width, numChannels)));

		}
		/*
		 * This is the base case for just one DriftChamber
		 */
		else if (modelType.equalsIgnoreCase("DriftChamber")) {

			temp = Models.DriftChamber(height, width, numChannels, numClasses, priors);
			setGridDimensions(temp);
			double[][] priorBoxes = FaultUtils.getPriors(priors, new double[][] {
					{ (double) width / (double) this.gridWidth, (double) height / (double) this.gridHeight } });// FaultUtils.allPriors

			this.computationGraph = Models.DriftChamber(height, width, numChannels, numClasses, priorBoxes);
			System.out.println(computationGraph.summary(InputType.convolutional(height, width, numChannels)));
			System.out.println(Arrays.deepToString(priors));
			System.out.println(Arrays.deepToString(priorBoxes));

		} else if (modelType.equalsIgnoreCase("Region")) {
			this.gridHeight = 36;
			this.gridWidth = 28;
			this.computationGraph = Models.RegionhModel(height, width, numChannels);
		} else if (modelType.equalsIgnoreCase("DCSystem")) {
			this.gridHeight = 36 * 3;
			this.gridWidth = 28;
			this.computationGraph = Models.CLASModel(height, width, numChannels);
		} else {
			throw new IllegalArgumentException("Invalid input: " + modelType);

		}
	}

	private void setGridDimensions(ComputationGraph computationGraph) {
		ComputationGraphConfiguration conf = computationGraph.getConfiguration();
		InputType[] inputTypes = { InputType.convolutional(height, width, numChannels) };
		GraphVertex[] vertices = computationGraph.getVertices();

		Map<String, InputType> vertexOutputs = new HashMap<>();
		int[] topologicalOrder = computationGraph.topologicalSortOrder();

		int currLayerIdx = -1;

		for (int currVertexIdx : topologicalOrder) {
			GraphVertex currentVertex = vertices[currVertexIdx];
			String currentVertexName = currentVertex.getVertexName();
			if (currentVertex.isInputVertex()) {
				if (inputTypes != null)
					vertexOutputs.put(currentVertexName,
							inputTypes[conf.getNetworkInputs().indexOf(currentVertexName)]);
			} else {
				List<InputType> inputTypeList = new ArrayList<>();
				if (currentVertex.hasLayer()) {
					if (inputTypes != null) {
						// get input type
						String inputVertexName = vertices[currentVertex.getInputVertices()[0].getVertexIndex()]
								.getVertexName();
						InputType currentInType = vertexOutputs.get(inputVertexName);
						inputTypeList.add(currentInType);
					}
					currLayerIdx++;
				} else {
					// get input type
					if (inputTypes != null) {
						VertexIndices[] inputVertices = currentVertex.getInputVertices();
						if (inputVertices != null) {
							for (int i = 0; i < inputVertices.length; i++) {
								GraphVertex thisInputVertex = vertices[inputVertices[i].getVertexIndex()];
								inputTypeList.add(vertexOutputs.get(thisInputVertex.getVertexName()));
							}
						}
					}
				}
				if (inputTypes != null) {
					InputType currentVertexOutputType = conf.getVertices().get(currentVertexName)
							.getOutputType(currLayerIdx, inputTypeList.toArray(new InputType[inputTypeList.size()]));
					vertexOutputs.put(currentVertexName, currentVertexOutputType);
				}
			}
		}

		List<GridDimensions> gridDimensions = new ArrayList<>();
		conf.getNetworkOutputs().forEach(k -> {
			int height = vertexOutputs.get(k).getShape()[1];
			int width = vertexOutputs.get(k).getShape()[2];
			gridDimensions.add(new GridDimensions(height, width));
		});
		this.gridHeight = gridDimensions.get(0).getHeight();
		this.gridWidth = gridDimensions.get(0).getWidth();
	}

	@Getter
	@AllArgsConstructor
	private class GridDimensions {
		private int height;
		private int width;

	}

	public static void main(String[] args) {
		CLASObject object = DriftChamber.builder().region(1).nchannels(1).maxFaults(10)
				.desiredFaults(Stream.of(FaultNames.CONNECTOR_TREE, FaultNames.CONNECTOR_TREE, FaultNames.CHANNEL_THREE,
						FaultNames.PIN_SMALL).collect(Collectors.toCollection(ArrayList::new)))
				.singleFaultGen(true).build();
		CLASModelFactory mFactory = new CLASModelFactory(object);
		System.out.println(mFactory.getGridWidth() + "  " + mFactory.getGridHeight());
		mFactory.getComputationGraph();
	}

}
