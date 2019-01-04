package utils;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.stream.Stream;
import java.util.zip.DataFormatException;

import org.bytedeco.javacv.CanvasFrame;
import org.datavec.image.data.Image;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.Upsampling2D;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.jlab.groot.base.ColorPalette;
import org.jlab.groot.base.TColorPalette;
import org.jlab.groot.data.H2F;
import org.jlab.groot.ui.TCanvas;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

public class FaultUtils {

	private FaultUtils() {
	}

	public static int RANGE_MAX = 200;
	public static int RANGE_MIN = 100;
	public static int FAULT_RANGE_MAX = 50;
	public static int FAULT_RANGE_MIN = 0;

	public static double[][] HVChannelPriors = { { 8.0, 6.0 }, { 16.0, 6.0 }, { 32.0, 6.0 } };
	public static double[][] HVPinPriors = { { 8.0, 1.0 }, { 16.0, 1.0 } };
	public static double[][] HVFusePriors = { { 6.0, 6.0 } };
	public static double[][] HVConnectorPriors = { { 4.0, 6.0 } };
	public static double[][] HVWirePriors = { { 1.0, 1.0 } };

	public static double[][] allPriors = merge(HVChannelPriors, HVPinPriors, HVFusePriors, HVConnectorPriors);
	public static double[][] allPriorsNoWire = merge(HVChannelPriors, HVPinPriors, HVFusePriors, HVConnectorPriors);

	public static double[][] merge(double[][]... arrays) {
		return Stream.of(arrays).flatMap(Stream::of).distinct() // or use
																// Arrays::stream
				.toArray(double[][]::new);
	}

	public static double[][] getPriors(double[][] scales) {
		double[][] ret = new double[allPriors.length][allPriors[0].length];
		for (int i = 0; i < allPriors.length; i++) {
			for (int j = 0; j < allPriors[0].length; j++) {
				ret[i][j] = allPriors[i][j] / scales[0][j];
			}
		}
		return ret;
	}

	public static double[][] getPriors(double[][] priors, double[][] scales) {
		double[][] ret = new double[priors.length][priors[0].length];
		for (int i = 0; i < priors.length; i++) {
			for (int j = 0; j < priors[0].length; j++) {
				// System.out.println(priors[i][j] + " " + scales[0][j] + " " +
				// priors[i][j] / scales[0][j]);
				ret[i][j] = priors[i][j] / scales[0][j];
			}
		}
		return ret;
	}

	public static double[][] getPriors(double[][] priors, double[][] scales, boolean switched) {
		double[][] ret = getPriors(priors, scales);
		if (switched) {
			double[][] arr = new double[priors[0].length][priors.length];
			for (int i = 0; i < arr.length; i++) {
				for (int j = 0; j < arr[0].length; j++) {
					// System.out.println(priors[i][j] + " " + scales[0][j] + "
					// " +
					// priors[i][j] / scales[0][j]);
					arr[i][j] = ret[j][i];
				}
			}
			double[][] arr2 = new double[priors.length][priors[0].length];
			for (int i = 0; i < arr2.length; i++) {
				for (int j = 0; j < arr2[0].length; j++) {
					// System.out.println(priors[i][j] + " " + scales[0][j] + "
					// " +
					// priors[i][j] / scales[0][j]);
					arr2[arr2.length - 1 - i][arr2[0].length - 1 - j] = arr[j][i];
				}
			}
			return arr2;
		} else {
			return ret;
		}
	}

	public static int[][] getData(int superLayer) throws DataFormatException {
		if (superLayer == 1) {
			return dataSector1;
		} else if (superLayer == 2) {
			return dataSector2;
		} else if (superLayer == 3) {
			return dataSector3;
		} else if (superLayer == 4) {
			return dataSector4;
		} else if (superLayer == 5) {
			return dataSector5;
		} else if (superLayer == 6) {
			return dataSector6;
		} else {
			System.err.println("Not A valid Sector");
			throw new DataFormatException();
		}
	}

	public static void draw(int[][] data) {
		TCanvas canvas = new TCanvas("Data", 800, 1200);
		canvas.getCanvas().setGridY(false);

		int x = data[0].length;
		int y = data.length;
		H2F hData = new H2F("Data", x, 1, x, y, 1, y);
		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				hData.setBinContent(i, j, data[j][i]);
			}
		}
		canvas.draw(hData);
	}

	public static void draw(double[][] data) {
		TCanvas canvas = new TCanvas("Data", 800, 1200);
		int x = data[0].length;
		int y = data.length;
		H2F hData = new H2F("Data", x, 1, x, y, 1, y);
		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				hData.setBinContent(i, j, data[j][i]);
			}
		}
		canvas.draw(hData);
	}

	/**
	 * 
	 * @param image
	 * @return displays the image of the INDArray
	 */
	public static void draw(Image image) {
		draw(image.getImage());
	}

	/**
	 * 
	 * @param image
	 * @param title
	 */
	public static void draw(Image image, String title) {
		draw(image.getImage(), title);
	}

	public static void draw(Image image, String title, String paletteType) {
		draw(image.getImage(), title, paletteType);
	}

	/**
	 * 
	 * @param image
	 * @return displays the image of the INDArray for <br>
	 *         specified int channel RGB - > 1,2,3
	 * 
	 */
	public static void draw(Image image, int channel) {
		draw(image.getImage(), channel);
	}

	/**
	 * 
	 * @param arr
	 * @return displays contents of INDArray
	 */
	public static void draw(INDArray arr) {
		draw(arr, "FU Plot");
	}

	/**
	 * 
	 * @param arr
	 *            input INDarray
	 * @parm Title of figure
	 * @return displays contents of INDArray
	 */
	public static void draw(INDArray arr, String title) {
		int rank = arr.rank();
		int rows = getRowsCols(arr)[0];
		int cols = getRowsCols(arr)[1];
		int nchannels = 1;
		if (rank != 2) {
			nchannels = (int) arr.size(rank == 3 ? 0 : 1);
		}
		double dataMin = (double) arr.minNumber();
		double dataMax = (double) arr.maxNumber();
		// System.out.println(rank + " rank from FaultUtils");

		BufferedImage b = new BufferedImage(cols, rows, BufferedImage.TYPE_INT_RGB);
		ColorPalette palette = new ColorPalette();
		System.out.println(arr.shapeInfoToString() + "  shape");
		if (nchannels == 1) {

			for (int y = 0; y < rows; y++) {
				for (int x = 0; x < cols; x++) {
					Color weightColor = null;
					if (rank == 2) {
						weightColor = palette.getColor3D(arr.getDouble(y, x), dataMax - dataMin, false);
					} else {
						weightColor = palette.getColor3D(rank == 3 ? arr.getDouble(0, y, x) : arr.getDouble(0, 0, y, x),
								dataMax - dataMin, false);
					}

					int red = weightColor.getRed();
					int green = weightColor.getGreen();
					int blue = weightColor.getBlue();
					int rgb = (red * 65536) + (green * 256) + blue;

					// b.setRGB(x, rows - y - 1, rgb);
					b.setRGB(x, y, rgb);

				}
			}
		} else if (nchannels == 3) {
			for (int y = 0; y < rows; y++) {
				for (int x = 0; x < cols; x++) {

					double red = rank == 3 ? arr.getDouble(0, y, x) : arr.getDouble(0, 0, y, x);
					double green = rank == 3 ? arr.getDouble(1, y, x) : arr.getDouble(0, 1, y, x);
					double blue = rank == 3 ? arr.getDouble(2, y, x) : arr.getDouble(0, 2, y, x);

					// double red = arr.getDouble(0, 0, y, x);
					// double green = arr.getDouble(0, 1, y, x);
					// double blue = arr.getDouble(0, 2, y, x);
					double rgb = ((red * 65536) + (green * 256) + blue);

					// b.setRGB(x, rows - y - 1, (int) rgb);
					b.setRGB(x, y, (int) rgb);

				}
			}
		} else {
			throw new ArithmeticException("Number of channels must be 1 or 3");
		}
		CanvasFrame cframe = new CanvasFrame("FaultUtils Plotted Me");
		cframe.setTitle(title);
		cframe.setCanvasSize(800, 600);
		cframe.showImage(b);
	}

	// "kDefault","kRainBow",
	// "kVisibleSpectrum","kDarkBodyRadiator","kInvertedDarkBodyRadiator"
	// /**
	// *
	// * @param arr
	// * INDarray: input INDArray title String: name of figure
	// * paletteType String: LUT Palette to draw;
	// *
	// "kDefault","kRainBow","kVisibleSpectrum","kDarkBodyRadiator","kInvertedDarkBodyRadiator"
	// *
	// * @return displays contents of INDArray
	// */
	/**
	 * 
	 * @param arr
	 *            INDarray: input INDArray
	 * @param title
	 *            String: name of figure
	 * @param paletteType
	 *            String: LUT Palette to draw;
	 *            "kDefault","kRainBow","kVisibleSpectrum","kDarkBodyRadiator","kInvertedDarkBodyRadiator"
	 * 
	 * @return displays contents of INDArray
	 */
	public static void draw(INDArray arr, String title, String paletteType) {
		int rank = arr.rank();
		int rows = getRowsCols(arr)[0];
		int cols = getRowsCols(arr)[1];
		int nchannels = 1;
		if (rank != 2) {
			nchannels = (int) arr.size(rank == 3 ? 0 : 1);
		}
		double dataMin = (double) arr.minNumber();
		double dataMax = (double) arr.maxNumber();
		// System.out.println(rank + " rank from FaultUtils");

		BufferedImage b = new BufferedImage(cols, rows, BufferedImage.TYPE_INT_RGB);
		TColorPalette palette = new TColorPalette();
		palette.setPalette(paletteType);
		System.out.println(arr.shapeInfoToString() + "  shape");
		if (nchannels == 1) {

			for (int y = 0; y < rows; y++) {
				for (int x = 0; x < cols; x++) {
					Color weightColor = null;
					if (rank == 2) {
						// weightColor = palette.getColor3D(arr.getDouble(y, x),
						// dataMax - dataMin, false);
						weightColor = palette.getColor3D(arr.getDouble(y, x), dataMin, dataMax, false);
					} else {
						// weightColor = palette.getColor3D(rank == 3 ?
						// arr.getDouble(0, y, x) : arr.getDouble(0, 0, y, x),
						// dataMax - dataMin, false);
						weightColor = palette.getColor3D(rank == 3 ? arr.getDouble(0, y, x) : arr.getDouble(0, 0, y, x),
								dataMin, dataMax, false);
					}

					int red = weightColor.getRed();
					int green = weightColor.getGreen();
					int blue = weightColor.getBlue();
					int rgb = (red * 65536) + (green * 256) + blue;

					// b.setRGB(x, rows - y - 1, rgb);
					b.setRGB(x, y, rgb);

				}
			}
		} else if (nchannels == 3) {
			for (int y = 0; y < rows; y++) {
				for (int x = 0; x < cols; x++) {

					double red = rank == 3 ? arr.getDouble(0, y, x) : arr.getDouble(0, 0, y, x);
					double green = rank == 3 ? arr.getDouble(1, y, x) : arr.getDouble(0, 1, y, x);
					double blue = rank == 3 ? arr.getDouble(2, y, x) : arr.getDouble(0, 2, y, x);

					// double red = arr.getDouble(0, 0, y, x);
					// double green = arr.getDouble(0, 1, y, x);
					// double blue = arr.getDouble(0, 2, y, x);
					double rgb = ((red * 65536) + (green * 256) + blue);

					// b.setRGB(x, rows - y - 1, (int) rgb);
					b.setRGB(x, y, (int) rgb);

				}
			}
		} else {
			throw new ArithmeticException("Number of channels must be 1 or 3");
		}
		CanvasFrame cframe = new CanvasFrame("FaultUtils Plotted Me");
		cframe.setTitle(title);
		cframe.setCanvasSize(800, 600);
		cframe.showImage(b);
	}

	/**
	 * 
	 * @param arr
	 * @return displays contents of INDArray for <br>
	 *         specified RGB -> 1,2,3
	 */

	public static void draw(INDArray arr, int channel) {
		draw(arr, channel, "FU Plot");
	}

	/**
	 * 
	 * @param arr
	 * @param String
	 *            title -> title of figure
	 * @return displays contents of INDArray for <br>
	 *         specified RGB -> 1,2,3
	 */
	public static void draw(INDArray arr, int channel, String title) {
		int rank = arr.rank();
		int nchannels = 1;
		if (rank != 2) {
			nchannels = (int) arr.size(rank == 3 ? 0 : 1);
		}
		if (nchannels == 1) {// regardless of input channel, plot the data
								// because there is not a color INDArray
			draw(arr, title);
		} else if (nchannels == 3) {

			int rows = getRowsCols(arr)[0];
			int cols = getRowsCols(arr)[1];

			BufferedImage b = new BufferedImage(cols, rows, BufferedImage.TYPE_INT_RGB);

			for (int y = 0; y < rows; y++) {
				for (int x = 0; x < cols; x++) {

					double red = rank == 3 ? arr.getDouble(0, y, x) : arr.getDouble(0, 0, y, x);
					double green = rank == 3 ? arr.getDouble(1, y, x) : arr.getDouble(0, 1, y, x);
					double blue = rank == 3 ? arr.getDouble(2, y, x) : arr.getDouble(0, 2, y, x);

					double rgb;
					// double rgb = ((red * 65536) + (green * 256) + blue);

					if (channel == 1) {
						// b.setRGB(x, rows - y - 1, (int) (red * 65536));
						b.setRGB(x, y, (int) (red * 65536));

					} else if (channel == 2) {
						// b.setRGB(x, rows - y - 1, (int) (green * 256));
						b.setRGB(x, y, (int) (green * 256));

					} else if (channel == 3) {
						// b.setRGB(x, rows - y - 1, (int) blue);
						b.setRGB(x, y, (int) blue);

					} else {
						throw new ArithmeticException("Can only draw channel 1, 2 or 3");
					}
				}
			}
			CanvasFrame cframe = new CanvasFrame("FaultUtils Plotted Me");
			cframe.setTitle(title);
			cframe.setCanvasSize(800, 600);
			cframe.showImage(b);
		} else {
			throw new ArithmeticException("Number of channels must be 1 or 3");
		}
	}

	public static INDArray toColor(int nChannels, INDArray input) {
		double dataMin = (double) input.minNumber();
		double dataMax = (double) input.maxNumber();
		int rank = input.rank();
		int rows = getRowsCols(input)[0];
		int cols = getRowsCols(input)[1];
		// System.out.println(nChannels + " ######### NCHANNELS");

		INDArray a = Nd4j.create(nChannels, rows, cols);
		ColorPalette palette = new ColorPalette();
		if (nChannels == 1) {
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					if (rank == 2) {
						a.putScalar(0, i, j, input.getDouble(i, j));
					} else {
						a.putScalar(0, i, j, rank == 3 ? input.getDouble(0, i, j) : input.getDouble(0, 0, i, j));
					}
				}
			}
		} else if (nChannels == 3) {
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					double value;
					if (rank == 2) {
						value = input.getDouble(i, j);
					} else {
						value = rank == 3 ? input.getDouble(0, i, j) : input.getDouble(0, 0, i, j);
					}

					Color weightColor = palette.getColor3D(value, dataMax - dataMin, false);
					int red = weightColor.getRed();
					int green = weightColor.getGreen();
					int blue = weightColor.getBlue();
					a.putScalar(0, i, j, red);
					a.putScalar(1, i, j, green);
					a.putScalar(2, i, j, blue);

				}
			}
		} else {
			throw new ArithmeticException("Number of channels must be 1 or 3");
		}
		return a;
	}

	/**
	 * 
	 * @param nChannels
	 *            Number of channels 1 or 3
	 * @param input
	 *            INDArray input
	 * @param paletteType
	 *            String: LUT Palette to draw;
	 *            "kDefault","kRainBow","kVisibleSpectrum","kDarkBodyRadiator","kInvertedDarkBodyRadiator"
	 * 
	 * @return displays contents of INDArray
	 */
	public static INDArray toColor(int nChannels, INDArray input, String paletteType) {
		double dataMin = (double) input.minNumber();
		double dataMax = (double) input.maxNumber();
		int rank = input.rank();
		int rows = getRowsCols(input)[0];
		int cols = getRowsCols(input)[1];
		// System.out.println(nChannels + " ######### NCHANNELS");

		INDArray a = Nd4j.create(nChannels, rows, cols);
		TColorPalette palette = new TColorPalette();
		palette.setPalette(paletteType);
		if (nChannels == 1) {
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					if (rank == 2) {
						a.putScalar(0, i, j, input.getDouble(i, j));
					} else {
						a.putScalar(0, i, j, rank == 3 ? input.getDouble(0, i, j) : input.getDouble(0, 0, i, j));
					}
				}
			}
		} else if (nChannels == 3) {
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					double value;
					if (rank == 2) {
						value = input.getDouble(i, j);
					} else {
						value = rank == 3 ? input.getDouble(0, i, j) : input.getDouble(0, 0, i, j);
					}

					Color weightColor = palette.getColor3D(value, dataMax - dataMin, false);
					int red = weightColor.getRed();
					int green = weightColor.getGreen();
					int blue = weightColor.getBlue();
					a.putScalar(0, i, j, red);
					a.putScalar(1, i, j, green);
					a.putScalar(2, i, j, blue);

				}
			}
		} else {
			throw new ArithmeticException("Number of channels must be 1 or 3");
		}
		return a;
	}

	public static Image asImageMatrix(int nChannels, INDArray data) {
		INDArray a = toColor(nChannels, data);

		Nd4j.getAffinityManager().tagLocation(a, AffinityManager.Location.HOST);
		a = a.reshape(ArrayUtil.combine(new long[] { 1 }, a.shape()));
		Image i = new Image(a, nChannels, data.rows(), data.columns());

		return i;

	}

	public static Image asImageMatrix(int dimensions, int nChannels, INDArray data) {
		INDArray a = toColor(nChannels, data);

		Nd4j.getAffinityManager().tagLocation(a, AffinityManager.Location.HOST);
		a = a.reshape(ArrayUtil.combine(new long[] { 1 }, a.shape()));
		// a = a.reshape('c', a.shape()[0], a.shape()[1], a.shape()[2],
		// a.shape()[3], a.shape()[4]);
		Image i = new Image(a, nChannels, data.rows(), data.columns());

		return i;

	}

	public static Image asUnShapedImageMatrix(int nChannels, INDArray data) {
		INDArray a = toColor(nChannels, data);

		Nd4j.getAffinityManager().tagLocation(a, AffinityManager.Location.HOST);
		// a = a.reshape(ArrayUtil.combine(new int[] { 1 }, a.shape()));
		Image i = new Image(a, nChannels, data.rows(), data.columns());

		return i;

	}

	public static Image asImage(int nChannels, INDArray input) {
		int rows = getRowsCols(input)[0];
		int cols = getRowsCols(input)[1];
		INDArray a = toColor(nChannels, input);
		Nd4j.getAffinityManager().tagLocation(a, AffinityManager.Location.HOST);
		// a = a.reshape(ArrayUtil.combine(new long[] { 1 }, a.shape()));
		// a = a.reshape('c', a.shape()[0], a.shape()[1], a.shape()[2],
		// a.shape()[3], a.shape()[4]);
		Image i = new Image(a, nChannels, rows, cols);

		return i;

		// int rows = getRowsCols(data)[0];
		// int cols = getRowsCols(data)[1];
		// Nd4j.getAffinityManager().tagLocation(data,
		// AffinityManager.Location.HOST);
		// Image i = new Image(data, nChannels, rows, cols);
		// return i;

	}

	public static int[] scaleImage(int preferredImageSize, int imageHeight, int imageWidth) {

		int heightScale = preferredImageSize / imageHeight;
		int widthScale = preferredImageSize / imageWidth;
		double heightRemainder = (double) preferredImageSize - (double) heightScale * imageHeight;
		double widthRemainder = (double) preferredImageSize - (double) widthScale * imageWidth;
		int heightPadding = (int) heightRemainder / 2;
		int widthPadding = (int) widthRemainder / 2;

		return new int[] { heightScale, widthScale, heightPadding, widthPadding };
	}

	public static int[] scaleImage(int[] preferredImageSize, int imageHeight, int imageWidth) {

		int heightScale = preferredImageSize[0] / imageHeight;
		int widthScale = preferredImageSize[1] / imageWidth;
		double heightRemainder = (double) preferredImageSize[0] - (double) heightScale * imageHeight;
		double widthRemainder = (double) preferredImageSize[1] - (double) widthScale * imageWidth;
		int heightPadding = (int) heightRemainder / 2;
		int widthPadding = (int) widthRemainder / 2;

		return new int[] { heightScale, widthScale, heightPadding, widthPadding };
	}

	public static int[] scaleImage(int imageHeight, int imageWidth) {
		return scaleImage(416, imageHeight, imageWidth);
	}

	public static INDArray scaleImage(INDArray input, int imageHeight, int imageWidth) {
		int[] scales = scaleImage(imageHeight, imageWidth);
		INDArray ret = FaultUtils.upSampleArray(input, new int[] { scales[0], scales[1] });
		return FaultUtils.zeroBorder(ret, scales[2], scales[3]);
	}

	public static INDArray scaleImage(INDArray input, int preferredImageSize, int imageHeight, int imageWidth) {
		int[] scales = scaleImage(preferredImageSize, imageHeight, imageWidth);
		INDArray ret = FaultUtils.upSampleArray(input, new int[] { scales[0], scales[1] });
		return FaultUtils.zeroBorder(ret, scales[2], scales[3]);
	}

	public static INDArray scaleImage(INDArray input, int[] preferredImageSize, int imageHeight, int imageWidth) {
		int[] scales = scaleImage(preferredImageSize, imageHeight, imageWidth);
		INDArray ret = FaultUtils.upSampleArray(input, new int[] { scales[0], scales[1] });
		return FaultUtils.zeroBorder(ret, scales[2], scales[3]);
	}

	public static INDArray zeroBorder(INDArray input) {
		return zeroBorder(input, 1, 1);
	}

	public static INDArray zeroBorder(INDArray input, int heightPad, int widthPad) {
		int rank = input.rank();
		int nchannels = (int) input.size(input.rank() == 3 ? 0 : 1);
		int rows = getRowsCols(input)[0];
		int cols = getRowsCols(input)[1];
		if (nchannels == 1) {
			input = Nd4j.prepend(input, widthPad, 0, input.rank() - 1);
			input = Nd4j.prepend(input, heightPad, 0, input.rank() - 2);
			input = Nd4j.append(input, widthPad, 0, input.rank() - 1);
			input = Nd4j.append(input, heightPad, 0, input.rank() - 2);
			return input;

		} else {
			throw new IllegalArgumentException("CHANNEL = 3 not yet supported");
		}
		/*
		 * ColorPalette palette = new ColorPalette(); // Find the minimum of the
		 * 3D matrix INDArray to1D = Nd4j.create(1, rows, cols); for (int y = 0;
		 * y < rows; y++) { for (int x = 0; x < cols; x++) { double red = rank
		 * == 3 ? input.getDouble(0, y, x) : input.getDouble(0, 0, y, x); double
		 * green = rank == 3 ? input.getDouble(1, y, x) : input.getDouble(0, 1,
		 * y, x); double blue = rank == 3 ? input.getDouble(2, y, x) :
		 * input.getDouble(0, 2, y, x); // b.setRGB(x, rows - y - 1, (int) (red
		 * * 65536));
		 * 
		 * // b.setRGB(x, rows - y - 1, (int) (green * 256));
		 * 
		 * // b.setRGB(x, rows - y - 1, (int) blue); System.out.print(((red /
		 * 65536) + (green / 256) + blue) + "  "); to1D.putScalar(0, y, x, (red
		 * / 65536) + (green / 256) + blue); } System.out.println(); }
		 * draw(to1D);
		 * 
		 * double dataMax = (double) to1D.maxNumber(); double dataMin = (double)
		 * to1D.minNumber(); // now get this color and the number for this color
		 * Color weightColor = palette.getColor3D(dataMin, dataMax - dataMin,
		 * false); int red = weightColor.getRed(); int green =
		 * weightColor.getGreen(); int blue = weightColor.getBlue(); // now put
		 * this into each slice INDArray slice = input.slice(0); // red INDArray
		 * redColors = slice.slice(0); redColors = Nd4j.prepend(redColors,
		 * widthPad, red, redColors.rank() - 1); redColors =
		 * Nd4j.prepend(redColors, heightPad, red, redColors.rank() - 2);
		 * redColors = Nd4j.append(redColors, widthPad, red, redColors.rank() -
		 * 1); redColors = Nd4j.append(redColors, heightPad, red,
		 * redColors.rank() - 2); // green INDArray greenColors =
		 * slice.slice(1); greenColors = Nd4j.prepend(greenColors, widthPad,
		 * green, greenColors.rank() - 1); greenColors =
		 * Nd4j.prepend(greenColors, heightPad, green, greenColors.rank() - 2);
		 * greenColors = Nd4j.append(greenColors, widthPad, green,
		 * greenColors.rank() - 1); greenColors = Nd4j.append(greenColors,
		 * heightPad, green, greenColors.rank() - 2); // blue INDArray
		 * blueColors = slice.slice(2); blueColors = Nd4j.prepend(blueColors,
		 * widthPad, blue, blueColors.rank() - 1); blueColors =
		 * Nd4j.prepend(blueColors, heightPad, blue, blueColors.rank() - 2);
		 * blueColors = Nd4j.append(blueColors, widthPad, blue,
		 * blueColors.rank() - 1); blueColors = Nd4j.append(blueColors,
		 * heightPad, blue, blueColors.rank() - 2);
		 * 
		 * // INDArray ret = Nd4j.create(3, rows, cols); INDArray a =
		 * Nd4j.accumulate(redColors, greenColors, blueColors); // INDArray a =
		 * Nd4j.hstack(redColors, greenColors); // a = Nd4j.hstack(a,
		 * blueColors);
		 * 
		 * Nd4j.getAffinityManager().tagLocation(a,
		 * AffinityManager.Location.HOST); a = a.reshape(ArrayUtil.combine(new
		 * long[] { 1 }, a.shape())); return a;
		 */

		// } else {
		// throw new ArithmeticException("Number of channels must be 1 or 3");
		// }

	}

	public static INDArray zeroXBorder(INDArray input) {
		input = Nd4j.prepend(input, 1, 0, input.rank() - 1);
		input = Nd4j.append(input, 1, 0, input.rank() - 1);
		return input;
	}

	/**
	 * 
	 * @param arr
	 * @return {nRows, nCols}
	 */
	public static int[] getRowsCols(INDArray arr) {
		int rank = arr.rank();
		if (rank == 2) {
			return new int[] { (int) arr.size(0), (int) arr.size(1) };
		} else {
			return new int[] { (int) arr.size(rank == 3 ? 1 : 2), (int) arr.size(rank == 3 ? 2 : 3) };
		}
	}

	public static H2F getHist(int[][] data) {
		H2F hData = new H2F("Data", 112, 1, 112, 6, 0, 6);
		for (int i = 0; i < data[0].length; i++) { // i are the rows
													// (layers)
			for (int j = 0; j < data.length; j++) { // j are the columns
													// (wires)
				hData.setBinContent(j, i, data[j][i]);
			}
		}
		return hData;
	}

	public static INDArray upSampleArray(INDArray input, int[] size) {
		return getUpsampling2DLayer(size).activate(input, false, LayerWorkspaceMgr.noWorkspaces());
	}

	public static INDArray downSampleArray(INDArray input, int[] kernalSize, int[] stride) {
		return getSubsamplingLayer(kernalSize, stride).activate(input, false, LayerWorkspaceMgr.noWorkspaces());
	}

	public static Layer getUpsampling2DLayer(int[] size) {
		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).seed(123)
				.layer(new Upsampling2D.Builder().size(size).build()).build();
		return conf.getLayer().instantiate(conf, null, 0, null, true);
	}

	public static Layer getSubsamplingLayer(int[] kernelSize, int[] stride) {
		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
				.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).seed(123)
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(kernelSize)
						.stride(stride).build())
				.build();

		return conf.getLayer().instantiate(conf, null, 0, null, true);
	}

	public static int[][] dataSector1 = { { 49, 49, 49, 29, 181, 135 }, { 100, 74, 334, 241, 611, 537 },
			{ 514, 433, 681, 675, 575, 584 }, { 637, 655, 610, 597, 715, 693 }, { 735, 656, 737, 705, 699, 673 },
			{ 767, 737, 724, 696, 693, 697 }, { 710, 716, 666, 698, 628, 637 }, { 658, 662, 628, 651, 595, 582 },
			{ 610, 615, 570, 586, 566, 560 }, { 586, 592, 563, 573, 530, 535 }, { 555, 526, 563, 527, 520, 514 },
			{ 552, 568, 525, 532, 536, 513 }, { 539, 534, 502, 534, 485, 488 }, { 487, 488, 479, 500, 465, 461 },
			{ 479, 486, 442, 463, 428, 456 }, { 459, 457, 459, 450, 456, 433 }, { 456, 463, 449, 455, 432, 425 },
			{ 460, 436, 450, 437, 411, 427 }, { 433, 421, 410, 420, 403, 391 }, { 410, 429, 389, 408, 386, 377 },
			{ 402, 396, 410, 403, 392, 382 }, { 382, 402, 409, 398, 400, 385 }, { 401, 396, 408, 401, 383, 381 },
			{ 384, 390, 383, 381, 389, 372 }, { 385, 383, 369, 384, 379, 367 }, { 382, 373, 406, 382, 371, 373 },
			{ 387, 377, 386, 372, 369, 369 }, { 401, 397, 405, 400, 386, 374 }, { 376, 408, 373, 387, 361, 373 },
			{ 295, 306, 339, 367, 343, 333 }, { 355, 360, 362, 346, 344, 343 }, { 355, 354, 354, 359, 349, 329 },
			{ 352, 365, 368, 383, 372, 367 }, { 371, 376, 383, 365, 362, 358 }, { 348, 377, 359, 366, 364, 350 },
			{ 351, 339, 367, 377, 361, 353 }, { 340, 349, 352, 357, 336, 334 }, { 340, 325, 358, 332, 363, 342 },
			{ 356, 376, 362, 375, 341, 357 }, { 326, 333, 366, 348, 347, 324 }, { 341, 349, 350, 340, 335, 332 },
			{ 322, 318, 332, 347, 348, 343 }, { 336, 339, 352, 339, 345, 340 }, { 340, 343, 331, 346, 330, 327 },
			{ 354, 342, 348, 337, 330, 334 }, { 319, 336, 339, 346, 319, 330 }, { 324, 328, 329, 335, 333, 322 },
			{ 320, 324, 325, 321, 319, 325 }, { 312, 319, 314, 321, 308, 296 }, { 306, 323, 321, 321, 318, 301 },
			{ 308, 319, 309, 321, 317, 289 }, { 282, 303, 301, 312, 299, 282 }, { 318, 319, 324, 325, 315, 295 },
			{ 326, 321, 338, 321, 310, 310 }, { 315, 327, 327, 321, 315, 298 }, { 309, 319, 308, 321, 300, 307 },
			{ 300, 311, 302, 305, 292, 291 }, { 296, 289, 298, 294, 314, 290 }, { 288, 307, 320, 318, 308, 307 },
			{ 310, 308, 316, 319, 295, 302 }, { 297, 313, 305, 295, 298, 294 }, { 308, 306, 306, 315, 305, 286 },
			{ 288, 305, 316, 299, 305, 304 }, { 287, 306, 291, 304, 302, 286 }, { 292, 278, 295, 292, 286, 268 },
			{ 272, 279, 283, 288, 270, 267 }, { 237, 276, 255, 276, 259, 268 }, { 251, 250, 276, 263, 268, 251 },
			{ 263, 275, 264, 269, 259, 252 }, { 251, 265, 277, 257, 256, 259 }, { 255, 264, 256, 270, 254, 248 },
			{ 253, 248, 251, 254, 256, 238 }, { 233, 241, 234, 238, 222, 227 }, { 225, 247, 240, 243, 228, 226 },
			{ 232, 243, 248, 245, 239, 235 }, { 218, 226, 226, 235, 234, 225 }, { 217, 224, 223, 221, 209, 212 },
			{ 204, 222, 222, 236, 218, 213 }, { 208, 206, 201, 216, 205, 225 }, { 187, 186, 184, 194, 186, 187 },
			{ 196, 202, 202, 199, 188, 196 }, { 183, 180, 183, 184, 185, 184 }, { 174, 178, 181, 186, 177, 179 },
			{ 168, 172, 162, 168, 164, 163 }, { 150, 169, 158, 163, 157, 162 }, { 156, 149, 153, 152, 154, 159 },
			{ 152, 154, 158, 161, 159, 152 }, { 142, 149, 144, 155, 145, 152 }, { 140, 142, 141, 140, 144, 140 },
			{ 125, 134, 133, 135, 122, 136 }, { 120, 124, 131, 132, 137, 119 }, { 116, 125, 127, 119, 115, 127 },
			{ 106, 114, 106, 115, 107, 114 }, { 96, 104, 102, 105, 90, 99 }, { 99, 97, 101, 104, 97, 93 },
			{ 86, 102, 91, 102, 93, 96 }, { 88, 96, 95, 95, 104, 96 }, { 80, 91, 95, 98, 89, 94 },
			{ 80, 81, 86, 82, 83, 80 }, { 70, 79, 82, 84, 80, 77 }, { 66, 75, 72, 78, 80, 75 },
			{ 54, 67, 63, 73, 67, 77 }, { 52, 57, 61, 56, 61, 59 }, { 45, 51, 48, 62, 50, 53 },
			{ 48, 48, 50, 40, 49, 48 }, { 40, 43, 47, 51, 46, 46 }, { 38, 43, 41, 46, 46, 44 },
			{ 40, 40, 40, 45, 43, 38 }, { 30, 38, 37, 37, 34, 38 }, { 30, 31, 28, 37, 35, 37 },
			{ 19, 27, 33, 25, 28, 33 }, { 8, 18, 12, 32, 20, 25 } };

	public static int[][] dataSector2 = { { 19, 19, 19, 12, 30, 24 }, { 32, 23, 73, 48, 267, 199 },
			{ 143, 110, 491, 370, 665, 612 }, { 625, 580, 625, 652, 605, 564 }, { 607, 614, 693, 632, 703, 684 },
			{ 753, 741, 689, 671, 661, 681 }, { 666, 680, 666, 690, 637, 622 }, { 672, 667, 631, 638, 606, 608 },
			{ 607, 619, 575, 602, 561, 549 }, { 560, 565, 540, 540, 518, 531 }, { 524, 517, 539, 508, 516, 522 },
			{ 519, 526, 522, 511, 501, 502 }, { 516, 522, 499, 515, 474, 478 }, { 462, 488, 458, 470, 444, 462 },
			{ 474, 464, 457, 462, 433, 455 }, { 467, 458, 446, 450, 429, 424 }, { 433, 450, 413, 424, 404, 403 },
			{ 421, 422, 420, 395, 408, 391 }, { 427, 408, 416, 429, 425, 410 }, { 421, 418, 414, 403, 371, 396 },
			{ 380, 403, 381, 402, 384, 382 }, { 388, 397, 408, 395, 392, 401 }, { 390, 394, 373, 399, 395, 373 },
			{ 370, 362, 348, 373, 373, 341 }, { 396, 381, 379, 388, 340, 366 }, { 355, 345, 366, 351, 364, 349 },
			{ 367, 356, 386, 367, 358, 363 }, { 358, 382, 388, 380, 363, 355 }, { 377, 373, 386, 384, 367, 364 },
			{ 373, 378, 374, 380, 368, 371 }, { 351, 368, 363, 380, 338, 330 }, { 338, 343, 354, 354, 350, 340 },
			{ 337, 350, 355, 345, 367, 357 }, { 364, 356, 381, 388, 372, 359 }, { 364, 389, 375, 364, 351, 358 },
			{ 344, 343, 345, 362, 347, 322 }, { 353, 354, 356, 358, 357, 343 }, { 354, 356, 361, 363, 351, 347 },
			{ 357, 358, 358, 363, 347, 355 }, { 345, 356, 348, 345, 356, 329 }, { 322, 328, 326, 361, 329, 341 },
			{ 335, 316, 348, 345, 338, 338 }, { 339, 339, 356, 333, 336, 315 }, { 347, 346, 319, 346, 342, 342 },
			{ 323, 337, 335, 342, 343, 336 }, { 341, 343, 360, 335, 336, 337 }, { 336, 348, 342, 353, 339, 332 },
			{ 330, 327, 316, 338, 335, 342 }, { 323, 322, 325, 310, 313, 315 }, { 311, 316, 325, 331, 321, 321 },
			{ 298, 329, 320, 327, 309, 317 }, { 310, 301, 318, 309, 315, 303 }, { 294, 321, 299, 311, 293, 299 },
			{ 334, 312, 338, 317, 326, 309 }, { 325, 329, 325, 339, 314, 329 }, { 296, 308, 318, 302, 308, 299 },
			{ 295, 305, 302, 299, 287, 297 }, { 303, 295, 302, 305, 301, 292 }, { 309, 306, 319, 319, 316, 320 },
			{ 324, 327, 316, 314, 320, 306 }, { 300, 313, 306, 308, 309, 303 }, { 296, 312, 305, 313, 305, 311 },
			{ 284, 286, 300, 292, 306, 282 }, { 283, 299, 291, 299, 288, 304 }, { 281, 301, 292, 288, 294, 299 },
			{ 273, 280, 274, 284, 269, 285 }, { 253, 272, 278, 274, 270, 259 }, { 264, 269, 252, 270, 255, 266 },
			{ 256, 265, 269, 263, 259, 258 }, { 252, 255, 262, 255, 259, 263 }, { 248, 256, 256, 254, 257, 249 },
			{ 246, 246, 246, 245, 246, 240 }, { 250, 260, 250, 248, 254, 250 }, { 241, 244, 239, 252, 234, 243 },
			{ 226, 237, 228, 232, 227, 223 }, { 210, 223, 214, 230, 217, 231 }, { 227, 220, 227, 222, 217, 213 },
			{ 233, 230, 230, 226, 223, 218 }, { 204, 228, 211, 218, 213, 215 }, { 201, 188, 200, 200, 195, 203 },
			{ 189, 197, 206, 206, 195, 194 }, { 185, 196, 183, 192, 191, 192 }, { 183, 184, 194, 191, 184, 187 },
			{ 165, 179, 182, 187, 179, 181 }, { 171, 161, 164, 168, 166, 172 }, { 146, 156, 154, 169, 151, 166 },
			{ 156, 156, 156, 150, 152, 147 }, { 155, 150, 149, 156, 150, 146 }, { 137, 146, 145, 149, 134, 145 },
			{ 122, 136, 131, 143, 134, 138 }, { 122, 125, 125, 133, 120, 126 }, { 121, 118, 120, 117, 117, 118 },
			{ 107, 122, 109, 110, 117, 119 }, { 104, 103, 108, 108, 111, 103 }, { 103, 104, 105, 106, 108, 107 },
			{ 88, 93, 95, 103, 98, 101 }, { 88, 94, 90, 94, 92, 100 }, { 79, 81, 80, 88, 86, 93 },
			{ 76, 83, 82, 77, 70, 82 }, { 76, 71, 73, 85, 76, 71 }, { 72, 81, 72, 72, 72, 73 },
			{ 66, 71, 73, 74, 75, 74 }, { 53, 68, 60, 71, 65, 69 }, { 55, 58, 57, 56, 58, 59 },
			{ 56, 55, 60, 50, 45, 57 }, { 44, 52, 51, 57, 49, 43 }, { 46, 44, 47, 48, 48, 49 },
			{ 50, 53, 51, 47, 47, 44 }, { 29, 42, 43, 44, 45, 47 }, { 31, 30, 30, 33, 32, 44 },
			{ 20, 34, 33, 35, 29, 33 }, { 5, 19, 10, 28, 24, 28 } };

	public static int[][] dataSector3 = { { 8, 8, 8, 5, 19, 16 }, { 16, 12, 31, 26, 80, 51 },
			{ 79, 43, 198, 119, 281, 245 }, { 323, 266, 343, 311, 377, 339 }, { 427, 372, 462, 400, 484, 469 },
			{ 556, 504, 540, 520, 561, 576 }, { 627, 585, 610, 598, 607, 626 }, { 608, 609, 552, 559, 575, 553 },
			{ 632, 573, 569, 584, 570, 613 }, { 594, 587, 565, 577, 557, 552 }, { 570, 558, 552, 535, 522, 532 },
			{ 501, 541, 473, 483, 470, 489 }, { 503, 485, 482, 474, 446, 471 }, { 446, 464, 443, 452, 405, 434 },
			{ 433, 421, 424, 430, 418, 409 }, { 420, 412, 388, 403, 395, 398 }, { 406, 400, 390, 377, 381, 403 },
			{ 390, 373, 381, 390, 371, 378 }, { 379, 371, 374, 371, 371, 378 }, { 363, 366, 351, 344, 362, 354 },
			{ 367, 348, 343, 361, 361, 358 }, { 363, 349, 349, 352, 344, 354 }, { 373, 362, 371, 367, 357, 365 },
			{ 361, 368, 357, 368, 345, 354 }, { 354, 365, 350, 343, 336, 352 }, { 356, 346, 366, 355, 359, 377 },
			{ 374, 355, 351, 354, 374, 363 }, { 365, 372, 376, 382, 370, 384 }, { 372, 364, 375, 368, 382, 384 },
			{ 375, 358, 372, 367, 368, 372 }, { 368, 364, 361, 368, 374, 357 }, { 369, 373, 388, 383, 372, 368 },
			{ 407, 382, 388, 387, 386, 374 }, { 382, 390, 392, 385, 384, 386 }, { 406, 393, 393, 388, 402, 387 },
			{ 413, 407, 392, 409, 407, 421 }, { 407, 403, 427, 394, 395, 391 }, { 397, 413, 401, 409, 399, 417 },
			{ 404, 408, 410, 400, 400, 392 }, { 412, 417, 412, 408, 419, 417 }, { 419, 411, 427, 424, 417, 420 },
			{ 424, 414, 421, 421, 423, 429 }, { 406, 428, 436, 431, 412, 427 }, { 423, 436, 425, 414, 426, 428 },
			{ 424, 410, 420, 417, 408, 426 }, { 423, 413, 417, 398, 408, 409 }, { 396, 413, 406, 409, 402, 402 },
			{ 402, 402, 395, 386, 395, 391 }, { 420, 404, 419, 414, 413, 411 }, { 404, 412, 408, 401, 397, 396 },
			{ 399, 404, 402, 392, 411, 403 }, { 399, 401, 404, 400, 395, 382 }, { 406, 401, 396, 395, 379, 386 },
			{ 392, 388, 388, 389, 385, 390 }, { 385, 391, 378, 373, 373, 380 }, { 377, 375, 373, 365, 359, 372 },
			{ 368, 359, 362, 359, 353, 342 }, { 386, 375, 359, 363, 347, 352 }, { 358, 357, 371, 361, 359, 362 },
			{ 337, 361, 349, 363, 351, 352 }, { 346, 347, 352, 338, 332, 344 }, { 351, 341, 343, 341, 336, 334 },
			{ 344, 348, 330, 332, 336, 325 }, { 324, 334, 338, 354, 337, 338 }, { 317, 326, 308, 320, 300, 324 },
			{ 314, 303, 299, 298, 292, 302 }, { 300, 292, 300, 299, 283, 302 }, { 299, 295, 295, 289, 286, 287 },
			{ 260, 280, 277, 289, 279, 290 }, { 261, 266, 251, 275, 258, 261 }, { 258, 268, 246, 268, 243, 260 },
			{ 259, 249, 252, 240, 242, 240 }, { 246, 257, 253, 247, 255, 248 }, { 247, 245, 245, 253, 249, 245 },
			{ 242, 254, 245, 246, 241, 238 }, { 227, 209, 233, 227, 231, 225 }, { 209, 228, 210, 218, 212, 218 },
			{ 215, 215, 225, 223, 215, 229 }, { 187, 204, 193, 211, 198, 209 }, { 195, 199, 194, 191, 182, 190 },
			{ 178, 193, 176, 181, 185, 178 }, { 168, 165, 173, 168, 166, 172 }, { 176, 175, 180, 176, 172, 168 },
			{ 152, 175, 152, 170, 157, 158 }, { 139, 150, 148, 143, 138, 154 }, { 131, 139, 132, 146, 137, 133 },
			{ 122, 124, 121, 118, 122, 124 }, { 121, 116, 111, 121, 114, 112 }, { 118, 121, 122, 116, 113, 113 },
			{ 108, 113, 107, 115, 107, 110 }, { 94, 103, 105, 106, 102, 112 }, { 90, 99, 95, 102, 102, 98 },
			{ 89, 98, 95, 97, 98, 96 }, { 86, 89, 83, 85, 89, 90 }, { 94, 89, 93, 83, 80, 81 },
			{ 82, 91, 89, 91, 92, 89 }, { 74, 81, 74, 84, 85, 84 }, { 72, 73, 71, 77, 72, 79 },
			{ 62, 66, 68, 70, 62, 62 }, { 65, 64, 63, 65, 64, 67 }, { 52, 61, 61, 57, 61, 60 },
			{ 51, 51, 54, 58, 54, 58 }, { 42, 48, 44, 49, 42, 51 }, { 40, 44, 47, 43, 45, 38 },
			{ 39, 46, 41, 45, 47, 44 }, { 35, 37, 36, 38, 35, 45 }, { 31, 32, 29, 36, 35, 34 },
			{ 25, 28, 27, 27, 26, 30 }, { 17, 23, 24, 27, 25, 25 }, { 15, 20, 22, 24, 21, 21 },
			{ 12, 17, 17, 20, 20, 24 }, { 3, 13, 7, 14, 10, 16 } };

	public static int[][] dataSector4 = { { 1, 19, 19, 22, 29, 29 }, { 20, 19, 35, 36, 35, 33 },
			{ 38, 38, 64, 44, 126, 91 }, { 145, 95, 219, 175, 257, 236 }, { 294, 259, 295, 272, 347, 325 },
			{ 391, 349, 424, 384, 439, 439 }, { 491, 445, 474, 472, 476, 485 }, { 540, 503, 501, 491, 505, 498 },
			{ 558, 514, 519, 523, 506, 504 }, { 551, 512, 485, 514, 475, 512 }, { 486, 461, 459, 458, 453, 472 },
			{ 520, 483, 486, 494, 493, 488 }, { 529, 491, 492, 467, 473, 504 }, { 446, 481, 455, 465, 419, 443 },
			{ 447, 430, 427, 427, 427, 447 }, { 422, 427, 427, 425, 441, 435 }, { 408, 403, 422, 399, 410, 417 },
			{ 397, 390, 376, 410, 380, 401 }, { 400, 383, 398, 381, 378, 385 }, { 387, 385, 378, 391, 390, 393 },
			{ 374, 365, 379, 398, 374, 394 }, { 378, 378, 371, 383, 375, 390 }, { 374, 367, 380, 388, 387, 397 },
			{ 377, 391, 380, 377, 380, 395 }, { 373, 365, 371, 377, 356, 344 }, { 366, 374, 374, 363, 387, 402 },
			{ 352, 371, 395, 342, 389, 389 }, { 366, 374, 377, 385, 378, 387 }, { 373, 379, 368, 372, 387, 391 },
			{ 378, 383, 377, 396, 384, 376 }, { 394, 380, 397, 377, 393, 401 }, { 389, 396, 411, 389, 412, 399 },
			{ 390, 402, 406, 406, 392, 404 }, { 401, 410, 394, 387, 401, 396 }, { 422, 414, 422, 400, 407, 417 },
			{ 387, 406, 408, 405, 425, 417 }, { 414, 401, 417, 422, 408, 408 }, { 407, 415, 403, 401, 397, 411 },
			{ 429, 414, 421, 414, 420, 431 }, { 414, 423, 428, 422, 422, 425 }, { 440, 444, 437, 430, 452, 425 },
			{ 431, 433, 443, 454, 446, 448 }, { 431, 438, 450, 447, 446, 443 }, { 429, 436, 420, 414, 413, 418 },
			{ 426, 424, 434, 418, 437, 425 }, { 417, 437, 416, 433, 421, 416 }, { 416, 405, 448, 431, 472, 425 },
			{ 433, 432, 434, 455, 421, 445 }, { 438, 421, 425, 423, 411, 439 }, { 410, 417, 405, 415, 397, 403 },
			{ 406, 400, 395, 385, 394, 392 }, { 401, 367, 391, 397, 394, 407 }, { 403, 408, 398, 390, 390, 383 },
			{ 389, 382, 385, 393, 385, 415 }, { 381, 382, 383, 382, 363, 394 }, { 382, 379, 373, 373, 378, 364 },
			{ 378, 375, 369, 386, 362, 364 }, { 365, 354, 356, 357, 349, 361 }, { 357, 352, 358, 351, 365, 366 },
			{ 357, 362, 354, 352, 347, 364 }, { 339, 343, 328, 338, 340, 338 }, { 319, 333, 347, 332, 332, 346 },
			{ 336, 323, 318, 334, 317, 334 }, { 321, 320, 310, 315, 317, 321 }, { 308, 317, 293, 303, 295, 305 },
			{ 296, 297, 291, 297, 282, 298 }, { 296, 290, 283, 284, 282, 282 }, { 275, 289, 282, 281, 287, 288 },
			{ 281, 275, 277, 286, 266, 274 }, { 260, 260, 255, 265, 254, 259 }, { 267, 261, 253, 262, 252, 260 },
			{ 277, 253, 263, 257, 255, 267 }, { 241, 264, 243, 250, 247, 261 }, { 247, 239, 238, 240, 234, 236 },
			{ 221, 232, 239, 227, 226, 231 }, { 220, 221, 223, 231, 221, 229 }, { 200, 213, 209, 222, 198, 222 },
			{ 207, 202, 193, 203, 210, 207 }, { 188, 193, 195, 194, 196, 195 }, { 191, 194, 199, 193, 190, 196 },
			{ 170, 170, 162, 183, 169, 185 }, { 159, 162, 162, 165, 162, 162 }, { 154, 153, 165, 153, 153, 158 },
			{ 155, 168, 141, 168, 156, 156 }, { 131, 145, 137, 144, 136, 153 }, { 128, 127, 127, 135, 132, 135 },
			{ 129, 129, 122, 125, 121, 128 }, { 108, 120, 117, 113, 111, 124 }, { 121, 111, 116, 123, 117, 114 },
			{ 109, 110, 106, 112, 111, 112 }, { 99, 107, 98, 105, 99, 107 }, { 88, 96, 94, 91, 90, 91 },
			{ 95, 84, 90, 93, 86, 88 }, { 82, 93, 85, 84, 82, 81 }, { 73, 76, 75, 80, 78, 78 },
			{ 74, 76, 71, 76, 77, 74 }, { 67, 77, 76, 77, 72, 74 }, { 68, 63, 61, 63, 61, 67 },
			{ 60, 67, 63, 63, 65, 61 }, { 51, 59, 60, 62, 62, 63 }, { 53, 56, 54, 57, 55, 54 },
			{ 51, 52, 54, 49, 49, 51 }, { 41, 46, 44, 50, 45, 49 }, { 38, 40, 38, 41, 38, 42 },
			{ 36, 42, 41, 39, 41, 36 }, { 29, 33, 33, 38, 38, 38 }, { 28, 25, 25, 28, 25, 32 },
			{ 26, 24, 24, 25, 24, 26 }, { 21, 23, 24, 23, 24, 21 }, { 21, 22, 17, 23, 24, 23 },
			{ 12, 21, 21, 23, 16, 19 }, { 3, 9, 7, 19, 18, 20 } };

	public static int[][] dataSector5 = { { 13, 13, 13, 10, 25, 19 }, { 18, 16, 26, 27, 49, 45 },
			{ 26, 24, 41, 43, 65, 71 }, { 42, 46, 73, 72, 87, 85 }, { 59, 62, 78, 84, 103, 105 },
			{ 70, 72, 103, 98, 114, 108 }, { 94, 87, 123, 122, 136, 119 }, { 144, 126, 192, 165, 206, 197 },
			{ 209, 196, 245, 226, 276, 247 }, { 299, 271, 343, 302, 356, 330 }, { 380, 360, 410, 380, 417, 413 },
			{ 448, 418, 449, 428, 442, 420 }, { 456, 460, 474, 453, 455, 464 }, { 506, 486, 496, 501, 484, 486 },
			{ 525, 514, 524, 494, 510, 507 }, { 510, 511, 513, 500, 522, 509 }, { 533, 525, 525, 529, 528, 512 },
			{ 493, 502, 512, 518, 492, 495 }, { 509, 500, 509, 506, 464, 495 }, { 512, 503, 514, 514, 520, 506 },
			{ 551, 502, 549, 545, 514, 530 }, { 505, 529, 517, 512, 495, 519 }, { 517, 499, 506, 497, 500, 505 },
			{ 536, 516, 531, 520, 515, 513 }, { 506, 501, 506, 512, 510, 510 }, { 542, 511, 526, 521, 498, 511 },
			{ 536, 534, 521, 510, 528, 517 }, { 509, 524, 505, 544, 489, 519 }, { 513, 503, 512, 491, 512, 481 },
			{ 526, 502, 505, 524, 508, 496 }, { 526, 515, 522, 514, 516, 507 }, { 507, 486, 517, 521, 499, 512 },
			{ 545, 533, 521, 514, 521, 498 }, { 511, 480, 488, 516, 478, 487 }, { 509, 512, 506, 494, 475, 465 },
			{ 476, 482, 445, 479, 469, 472 }, { 495, 463, 463, 451, 462, 440 }, { 505, 466, 504, 490, 507, 490 },
			{ 477, 492, 479, 489, 459, 467 }, { 482, 464, 461, 447, 460, 468 }, { 483, 497, 470, 453, 458, 449 },
			{ 458, 448, 462, 469, 447, 443 }, { 460, 457, 455, 447, 450, 452 }, { 467, 451, 444, 452, 444, 441 },
			{ 458, 459, 458, 440, 435, 444 }, { 431, 451, 456, 441, 432, 448 }, { 448, 438, 438, 436, 417, 431 },
			{ 421, 440, 445, 427, 435, 429 }, { 474, 444, 437, 436, 420, 415 }, { 399, 433, 423, 422, 407, 418 },
			{ 411, 400, 395, 399, 408, 407 }, { 406, 411, 411, 408, 403, 405 }, { 411, 408, 396, 393, 392, 394 },
			{ 377, 407, 385, 391, 386, 405 }, { 391, 391, 387, 391, 363, 387 }, { 370, 391, 381, 367, 372, 370 },
			{ 369, 365, 363, 376, 351, 378 }, { 346, 354, 365, 358, 340, 336 }, { 348, 358, 348, 351, 359, 352 },
			{ 332, 350, 334, 344, 337, 333 }, { 335, 338, 331, 322, 327, 320 }, { 301, 340, 326, 323, 303, 308 },
			{ 313, 302, 293, 303, 299, 299 }, { 286, 296, 307, 307, 293, 294 }, { 285, 292, 296, 292, 285, 293 },
			{ 289, 289, 286, 289, 276, 265 }, { 268, 287, 279, 284, 275, 271 }, { 260, 270, 269, 268, 268, 256 },
			{ 242, 259, 252, 256, 248, 249 }, { 241, 259, 258, 241, 237, 231 }, { 231, 241, 233, 251, 234, 235 },
			{ 212, 234, 226, 233, 214, 225 }, { 210, 218, 212, 200, 205, 202 }, { 199, 215, 215, 207, 196, 198 },
			{ 193, 209, 191, 203, 185, 182 }, { 177, 189, 182, 181, 186, 183 }, { 171, 184, 179, 190, 168, 177 },
			{ 158, 176, 171, 169, 172, 157 }, { 129, 167, 165, 173, 161, 151 }, { 140, 142, 148, 158, 150, 157 },
			{ 121, 149, 148, 154, 136, 144 }, { 115, 123, 114, 131, 134, 130 }, { 107, 128, 111, 123, 112, 129 },
			{ 84, 112, 116, 114, 111, 105 }, { 90, 99, 109, 113, 112, 101 }, { 84, 88, 91, 105, 97, 100 },
			{ 79, 94, 87, 91, 85, 90 }, { 73, 85, 81, 86, 87, 76 }, { 68, 75, 81, 82, 80, 74 },
			{ 61, 73, 64, 81, 79, 77 }, { 54, 67, 66, 64, 65, 70 }, { 52, 59, 57, 58, 55, 55 },
			{ 48, 60, 58, 55, 59, 55 }, { 50, 45, 47, 57, 50, 52 }, { 40, 52, 57, 46, 49, 45 },
			{ 36, 46, 40, 52, 52, 46 }, { 35, 38, 42, 42, 41, 39 }, { 33, 37, 38, 40, 42, 36 },
			{ 27, 35, 35, 33, 39, 36 }, { 24, 30, 32, 35, 33, 34 }, { 26, 28, 27, 30, 27, 28 },
			{ 23, 29, 28, 26, 25, 25 }, { 23, 23, 25, 27, 30, 22 }, { 16, 23, 23, 26, 25, 25 },
			{ 13, 17, 17, 19, 19, 22 }, { 10, 15, 12, 21, 16, 14 }, { 12, 14, 17, 15, 16, 14 },
			{ 11, 12, 9, 14, 13, 15 }, { 9, 14, 14, 10, 12, 11 }, { 4, 10, 10, 14, 11, 10 }, { 3, 6, 7, 8, 10, 12 },
			{ 1, 4, 4, 9, 8, 9 } };

	public static int[][] dataSector6 = { { 2, 1, 58, 61, 90, 85 }, { 39, 46, 82, 89, 89, 88 },
			{ 79, 77, 104, 91, 91, 86 }, { 84, 92, 105, 106, 86, 86 }, { 92, 96, 105, 94, 105, 98 },
			{ 101, 97, 112, 113, 105, 101 }, { 104, 109, 122, 113, 121, 112 }, { 112, 106, 154, 140, 162, 150 },
			{ 184, 168, 213, 196, 244, 217 }, { 237, 217, 281, 272, 281, 263 }, { 333, 309, 347, 335, 350, 329 },
			{ 371, 348, 415, 376, 412, 382 }, { 442, 422, 450, 441, 448, 440 }, { 477, 458, 458, 450, 475, 462 },
			{ 466, 457, 503, 482, 482, 479 }, { 499, 476, 507, 487, 482, 479 }, { 521, 498, 517, 502, 507, 513 },
			{ 517, 509, 512, 516, 501, 488 }, { 507, 515, 516, 505, 477, 499 }, { 505, 492, 458, 493, 473, 509 },
			{ 468, 456, 487, 475, 461, 459 }, { 507, 472, 489, 477, 467, 467 }, { 520, 483, 488, 484, 474, 477 },
			{ 502, 490, 485, 492, 464, 474 }, { 488, 483, 494, 492, 492, 481 }, { 510, 488, 492, 475, 459, 478 },
			{ 517, 492, 493, 479, 507, 476 }, { 492, 497, 523, 524, 494, 509 }, { 511, 515, 492, 500, 467, 480 },
			{ 504, 481, 487, 466, 480, 495 }, { 508, 482, 496, 514, 481, 487 }, { 514, 486, 482, 475, 468, 465 },
			{ 490, 487, 484, 495, 479, 508 }, { 488, 490, 481, 480, 490, 471 }, { 470, 471, 455, 476, 449, 478 },
			{ 462, 466, 460, 453, 480, 467 }, { 442, 461, 444, 457, 456, 483 }, { 444, 419, 457, 444, 419, 458 },
			{ 474, 460, 435, 438, 430, 429 }, { 429, 435, 430, 432, 422, 431 }, { 415, 420, 421, 396, 404, 434 },
			{ 434, 409, 416, 414, 400, 409 }, { 435, 435, 439, 428, 419, 412 }, { 420, 437, 412, 433, 395, 403 },
			{ 415, 415, 414, 413, 412, 428 }, { 432, 425, 417, 423, 422, 407 }, { 419, 402, 427, 392, 411, 410 },
			{ 422, 417, 410, 423, 413, 417 }, { 414, 404, 395, 387, 391, 400 }, { 418, 393, 400, 390, 376, 389 },
			{ 401, 401, 368, 371, 377, 392 }, { 408, 385, 377, 390, 376, 398 }, { 384, 378, 370, 366, 360, 371 },
			{ 375, 356, 373, 360, 357, 374 }, { 384, 362, 370, 360, 353, 358 }, { 363, 363, 368, 370, 364, 367 },
			{ 349, 363, 338, 354, 343, 352 }, { 335, 352, 338, 347, 317, 344 }, { 325, 323, 308, 313, 322, 328 },
			{ 311, 310, 293, 311, 302, 315 }, { 297, 292, 290, 269, 298, 304 }, { 301, 299, 299, 294, 275, 278 },
			{ 277, 270, 273, 287, 288, 273 }, { 265, 273, 275, 281, 279, 277 }, { 272, 266, 262, 268, 245, 273 },
			{ 258, 262, 254, 259, 267, 266 }, { 237, 259, 240, 241, 244, 237 }, { 236, 234, 238, 243, 246, 245 },
			{ 243, 256, 240, 248, 233, 236 }, { 217, 232, 231, 232, 209, 214 }, { 209, 216, 204, 220, 207, 202 },
			{ 196, 212, 217, 214, 214, 208 }, { 193, 200, 205, 207, 194, 201 }, { 166, 194, 180, 195, 182, 174 },
			{ 173, 171, 176, 176, 176, 176 }, { 157, 166, 163, 167, 165, 164 }, { 156, 167, 167, 172, 159, 155 },
			{ 153, 171, 156, 171, 169, 156 }, { 144, 156, 160, 158, 148, 151 }, { 127, 140, 136, 148, 139, 146 },
			{ 121, 132, 131, 129, 130, 125 }, { 107, 129, 125, 133, 119, 130 }, { 108, 113, 119, 124, 114, 110 },
			{ 97, 111, 112, 116, 112, 117 }, { 85, 93, 90, 106, 100, 99 }, { 77, 96, 97, 103, 99, 95 },
			{ 69, 91, 89, 95, 92, 90 }, { 62, 73, 75, 84, 79, 84 }, { 67, 68, 70, 72, 72, 76 },
			{ 59, 73, 75, 65, 63, 61 }, { 48, 64, 66, 76, 65, 69 }, { 54, 61, 65, 68, 62, 64 },
			{ 49, 59, 63, 60, 64, 55 }, { 37, 50, 49, 59, 59, 55 }, { 31, 44, 45, 51, 49, 52 },
			{ 42, 43, 45, 46, 44, 49 }, { 37, 42, 41, 42, 43, 39 }, { 28, 40, 43, 42, 42, 35 },
			{ 31, 31, 36, 41, 41, 39 }, { 26, 36, 38, 37, 32, 32 }, { 27, 35, 35, 36, 40, 27 },
			{ 20, 29, 29, 31, 33, 32 }, { 19, 27, 28, 30, 29, 27 }, { 18, 28, 28, 27, 25, 23 },
			{ 15, 23, 21, 26, 27, 22 }, { 14, 21, 23, 25, 19, 25 }, { 13, 20, 19, 22, 20, 17 },
			{ 10, 19, 19, 14, 17, 16 }, { 13, 14, 16, 19, 17, 14 }, { 8, 14, 15, 14, 18, 15 },
			{ 2, 10, 10, 15, 13, 12 }, { 1, 1, 1, 12, 14, 11 } };

	public static double[][] convertToDouble(int[][] data) {
		double[][] retValue = new double[data.length][data[0].length];
		for (int i = 0; i < data[0].length; i++) {
			for (int j = 0; j < data.length; j++) {
				retValue[j][i] = (double) data[j][i];
			}
		}
		return retValue;
	}

	public static void main(String[] args) {

		int[] tempscale = FaultUtils.scaleImage(6, 112);

		double[][] scales = { { 1.0 / (double) tempscale[1], 1.0 / (double) tempscale[0] } };

		double[][] priors = allPriors;
		double[][] test = getPriors(priors,
				new double[][] { { (double) (1.0 / (double) (416 / 112)), 1.0 / (double) (416 / 6) } });
		double[][] testII = priors;
		testII = FaultUtils.getPriors(testII, scales);

		System.out.println(Arrays.deepToString(allPriors));
		System.out.println(Arrays.deepToString(test));
		System.out.println(Arrays.deepToString(testII));

	}
}
