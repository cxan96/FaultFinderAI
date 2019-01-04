package processHipo;

import java.util.ArrayList;
import java.util.List;

import org.jlab.groot.data.H2F;
import org.jlab.groot.ui.TCanvas;
import org.nd4j.linalg.api.ndarray.INDArray;

import lombok.Getter;

public class FaultDataContainer {

	private List<Container> aContainers = null;

	public FaultDataContainer() {
		aContainers = new ArrayList<>();
		createList();
	}

	private void createList() {
		for (int i = 1; i < 7; i++) {
			for (int j = 1; j < 7; j++) {
				aContainers.add(new Container(i, j));
			}
		}
	}

	public List<Container> getList() {
		return aContainers;
	}

	public void increment(int sector, int superLayer, int wire, int layer) {
		aContainers.get(aContainers.indexOf(new Container(sector, superLayer))).increment(wire, layer);
	}

	public int[][] getData(int sector, int superLayer) {
		return aContainers.get(aContainers.indexOf(new Container(sector, superLayer))).getData();
	}

	public void plotData() {
		for (int i = 1; i < 7; i++) {
			for (int j = 1; j < 7; j++) {
				int[][] data = getData(i, j);
				TCanvas canvas = new TCanvas("Training Data", 800, 1200);
				H2F hData = new H2F("Training Data", 112, 1, 112, 6, 1, 6);
				for (int k = 0; k < data[0].length; k++) { // k are the rows
															// (layers)
					for (int l = 0; l < data.length; l++) { // l are the columns
															// (wires)
						hData.setBinContent(k, l, data[l][k]);
					}
				}
				canvas.draw(hData);
			}
		}
	}

	public void plotData(int sector, int superLayer) {
		int[][] data = getData(sector, superLayer);
		TCanvas canvas = new TCanvas("Training Data S: " + sector + " SL: " + superLayer, 800, 1200);
		H2F hData = new H2F("Training Data", 112, 1, 112, 6, 1, 6);
		for (int k = 0; k < data[0].length; k++) { // k are the rows
													// (layers)
			for (int l = 0; l < data.length; l++) { // l are the columns
													// (wires)
				hData.setBinContent(k, l, data[l][k]);
			}
		}
		canvas.draw(hData);
	}

	public void plotData(int sector, int superLayer, boolean withGrid) {
		int[][] data = getData(sector, superLayer);
		TCanvas canvas = new TCanvas("Training Data S: " + sector + " SL: " + superLayer, 800, 1200);
		H2F hData = new H2F("Training Data", 112, 1, 112, 6, 1, 6);
		for (int k = 0; k < data[0].length; k++) { // k are the rows
													// (layers)
			for (int l = 0; l < data.length; l++) { // l are the columns
													// (wires)
				hData.setBinContent(k, l, data[l][k]);
			}
		}
		canvas.draw(hData);
	}

	class Container {

		@Getter
		private int sector;
		@Getter
		private int superLayer;
		@Getter
		private int[][] data;
		private INDArray features;

		public Container(int sector, int superLayer) {
			this.sector = sector;
			this.superLayer = superLayer;
			this.data = new int[6][112];

		}

		public Container(int sector, int superLayer, INDArray features) {
			this.sector = sector;
			this.superLayer = superLayer;
			this.features = features;

		}

		public void increment(int wire, int layer) {
			data[layer - 1][wire - 1]++;
		}

		public INDArray getFeatures() {
			return this.features;

		}

		private FaultDataContainer getOuterType() {
			return FaultDataContainer.this;
		}

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + getOuterType().hashCode();
			result = prime * result + sector;
			result = prime * result + superLayer;
			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			Container other = (Container) obj;
			if (!getOuterType().equals(other.getOuterType()))
				return false;
			if (sector != other.sector)
				return false;
			if (superLayer != other.superLayer)
				return false;
			return true;
		}
	}

	public static void main(String[] args) {
		FaultDataContainer faultDataContainer = new FaultDataContainer();
		List<Container> aList = faultDataContainer.getList();
		System.out.println(faultDataContainer.getList().size());
		int sector = aList.get(0).getSector();
		int superLayer = aList.get(0).getSuperLayer();
		System.out.println(sector + "  " + superLayer);
		faultDataContainer.increment(6, 1, 33, 1);

	}
}
