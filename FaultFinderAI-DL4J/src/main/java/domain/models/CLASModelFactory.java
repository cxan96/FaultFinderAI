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
import clasDC.faults.FaultSlidingInformation;
import clasDC.objects.CLASObject;
import clasDC.objects.CLASObject.ContainerType;
import clasDC.objects.SuperLayer;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

/**
 * @author m.c.kunkel
 *
 */
public class CLASModelFactory {

	private int height;
	private int width;
	private int numChannels;
	private int numClasses;
	private double[][] priors;
	private String modelType;
	private FaultNames desiredFault;
	private ContainerType containerType;
	private CLASObject clasObject;
	@Getter
	private int gridWidth;
	@Getter
	private int gridHeight;
	@Getter
	ComputationGraph computationGraph = null;

	private InputType inputType = null;

	@Builder
	public CLASModelFactory(CLASObject clasObject) {
		this.numChannels = clasObject.getNchannels();
		this.priors = clasObject.getPriors();
		this.modelType = clasObject.getObjectType();
		this.desiredFault = clasObject.getDesiredFault();
		this.clasObject = clasObject;
		this.containerType = clasObject.getContainerType();
		init();
		setModel();
	}

	/**
	 * init() will set the correct height, width, numclasses depending on the
	 * CLASObject+ContainerType+DesiredFault
	 * 
	 * THIS NEEDS TO BE FIXED BECAUSE numclasses is set in CLASObject <br>
	 * Fix one or the other
	 */
	private void init() {
		if (this.containerType.equals(ContainerType.SEG)) {
			this.height = this.clasObject.getHeight();
			this.width = this.clasObject.getWidth();
			this.numClasses = this.clasObject.getDesiredFaults().stream().distinct().collect(Collectors.toList())
					.size();

		} else if (this.containerType.equals(ContainerType.CLASS)) {
			this.height = this.clasObject.getHeight();
			this.width = this.clasObject.getWidth();
			this.numClasses = 2;

		} else if (this.containerType.equals(ContainerType.OBJ)) {
			if (this.desiredFault == null) {
				throw new IllegalArgumentException("Cannot perform object detection without a desired fault");
			}
			this.width = this.desiredFault.getFSlidingInformation().getXEnd()[0]
					- this.desiredFault.getFSlidingInformation().getXStart()[0];
			if (this.desiredFault.equals(FaultNames.PIN_SMALL) || this.desiredFault.equals(FaultNames.PIN_BIG)
					|| this.desiredFault.equals(FaultNames.DEADWIRE) || this.desiredFault.equals(FaultNames.HOTWIRE)) {
				this.height = 2;
				this.numClasses = 3;
			} else {
				this.height = 6;
				this.numClasses = 2;
			}
		} else if (this.containerType.equals(ContainerType.MULTICLASS)) {
			FaultSlidingInformation faultSlidingInformation = this.desiredFault.getFSlidingInformation();
			this.height = this.clasObject.getHeight();
			// if (this.desiredFault.equals(FaultNames.CHANNEL_ONE)) {
			// this.width = faultSlidingInformation.getXStart()[0] + 18;
			// this.numClasses = 3;
			// } else {

			this.width = faultSlidingInformation.getXEnd()[faultSlidingInformation.getXLength() - 1]
					- faultSlidingInformation.getXStart()[0];
			this.numClasses = (int) this.desiredFault.getPossiblePositions().length();
			System.out.println(this.height + "  " + this.width);

		} else {
			throw new IllegalArgumentException("The container type is not recognized " + containerType);
		}

		this.inputType = InputType.convolutional(height, width, numChannels);

	}

	private void setModel() {
		if (containerType.equals(ContainerType.CLASS)) {
			System.out.println("The model is set to Classification Model");
			setClassificationModel();
		} else if (containerType.equals(ContainerType.OBJ)) {
			System.out.println("The model is set to Object Detection Model");
			setObjectModel();
		} else if (containerType.equals(ContainerType.MULTICLASS)) {
			System.out.println("The model is set to MultiClass Model");
			setMultiClassModel();
		} else if (containerType.equals(ContainerType.SEG)) {
			System.out.println("The model is set to Segmentation Model");
			setSegmentationModel();
		} else {
			throw new IllegalArgumentException("The container type is not recognized " + containerType);
		}
	}

	private void setClassificationModel() {
		this.computationGraph = Models.KunkelPeters(height, width, numChannels, numClasses).toComputationGraph();
	}

	private void setObjectModel() {
		/*
		 * This is the base case for just one superlayer
		 */
		if (modelType.equalsIgnoreCase("SuperLayer")) {
			// PIN_SMALL,PIN_BIG,DEADWIRE, HOTWIRE all use multilabel
			// classification using regresssion
			if (this.desiredFault.equals(FaultNames.PIN_SMALL) || this.desiredFault.equals(FaultNames.PIN_BIG)
					|| this.desiredFault.equals(FaultNames.DEADWIRE) || this.desiredFault.equals(FaultNames.HOTWIRE)) {

				this.computationGraph = Models.deeperDNNMultiClass(height, width, numChannels, numClasses)
						.toComputationGraph();
			} else {
				this.computationGraph = Models.KunkelPeters(height, width, numChannels, numClasses)
						.toComputationGraph();
			}

		} else {
			throw new IllegalArgumentException("Invalid input: " + modelType);

		}
	}

	private void setSegmentationModel() {
		/*
		 * This is the base case for just one superlayer
		 */
		if (modelType.equalsIgnoreCase("SuperLayer")) {
			// this.computationGraph = SegmentationModels.UFault(252, 252, 1);
			this.computationGraph = SegmentationModels.KunkelPetersUYolo4SL(8, 116, numChannels);
			System.out.println(computationGraph.summary(InputType.convolutional(8, 116, numChannels)));

		}
	}

	private void setMultiClassModel() {
		/*
		 * This is the base case for just one superlayer
		 */
		if (modelType.equalsIgnoreCase("SuperLayer")) {
			if (this.desiredFault.equals(FaultNames.DEADWIRE) || this.desiredFault.equals(FaultNames.HOTWIRE)) {
				this.computationGraph = Models.wireMultiClass(height, width, numChannels, numClasses)
						.toComputationGraph();
			} else {
				this.computationGraph = Models.deeperDNNMultiClass(height, width, numChannels, numClasses)
						.toComputationGraph();
			}
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
			int height = (int) vertexOutputs.get(k).getShape()[1];
			int width = (int) vertexOutputs.get(k).getShape()[2];
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

	private void disclosure(double[][] priorBoxes) {
		System.out.println("###############################################");
		System.out.println("############# Disclosure of model #############");
		System.out.println("This model parameters are ");
		System.out.println(computationGraph.summary(inputType));
		System.out.println("###############################################");
		System.out.println(
				"With numClasses  = " + numClasses + " and bounding boxes set to " + Arrays.deepToString(priorBoxes));
		System.out.println("###############################################");
		System.out.println("###############################################");

	}

	public static void main(String[] args) {
		CLASObject object = SuperLayer.builder().superlayer(1).nchannels(1).minFaults(8).maxFaults(10)
				.desiredFault(FaultNames.PIN_BIG)
				.desiredFaults(
						Stream.of(FaultNames.CONNECTOR_THREE, FaultNames.CONNECTOR_TREE, FaultNames.CHANNEL_THREE,
								FaultNames.PIN_SMALL).collect(Collectors.toCollection(ArrayList::new)))
				.singleFaultGen(false).isScaled(true).containerType(ContainerType.OBJ).build();
		CLASModelFactory mFactory = new CLASModelFactory(object);
		System.out.println(mFactory.getGridWidth() + "  " + mFactory.getGridHeight());
		mFactory.getComputationGraph();
		// System.out.println(Arrays.deepToString(object.getPriors()));
		//
		// List<FaultNames> aList =
		// object.getDesiredFaults().stream().distinct().collect(Collectors.toList());
		// for (FaultNames fault : aList) {
		// System.out.println(fault.getSaveName());
		// }
		// int numClasses =
		// object.getDesiredFaults().stream().distinct().collect(Collectors.toList()).size();
		// System.out.println(numClasses);
	}

}
