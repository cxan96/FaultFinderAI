package processHipo;

import java.util.ArrayList;
import java.util.List;

import org.jlab.io.base.DataBank;
import org.jlab.io.base.DataEvent;
import org.jlab.io.hipo.HipoDataSource;

public class DataProcess {
	private HipoDataSource reader = null;
	private List<String> fileList = null;

	private FaultDataContainer fContainer = null;

	public DataProcess() {
		init();
	}

	public DataProcess(String... strings) {
		this.fileList = new ArrayList<>();
		createFileList(strings);
		init();
	}

	public DataProcess(List<String> fileList) {
		this.fileList = fileList;
		init();
	}

	private void init() {
		this.fContainer = new FaultDataContainer();
	}

	public void setFileList(List<String> list) {
		this.fileList = list;
	}

	private void createFileList(String... strings) {
		for (String string : strings) {
			this.fileList.add(string);
		}
	}

	public void processFile() {
		for (String string : fileList) {
			this.reader = new HipoDataSource();
			this.reader.open(string);
			fillData();
			this.reader.close();
		}
	}

	public void plotData() {
		this.fContainer.plotData();
	}

	private void fillData() {
		int counter = 0;

		while (this.reader.hasEvent()) {// && counter < 400 &&
										// counter < nEvents
			if (counter % 10000 == 0) {
				System.out.println("done " + counter + " events");
			}
			DataEvent event = reader.getNextEvent();
			counter++;
			if (event.hasBank("TimeBasedTrkg::TBHits")) {
				processTBHits(event);
			}
		}
	}

	private void processTBHits(DataEvent event) {
		DataBank bnkHits = event.getBank("TimeBasedTrkg::TBHits");
		for (int i = 0; i < bnkHits.rows(); i++) {

			fContainer.increment(bnkHits.getInt("sector", i), bnkHits.getInt("superlayer", i),
					bnkHits.getInt("wire", i), bnkHits.getInt("layer", i));

		}
	}

	public int[][] getData(int sector, int superLayer) {
		return fContainer.getData(sector, superLayer);
	}

}
