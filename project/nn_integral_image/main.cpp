/*
 *  TU Eindhoven
 *  Eindhoven, The Netherlands
 *
 *  Name            :   faceDetection.cpp
 *
 *  Author          :   Francesco Comaschi (f.comaschi@tue.nl)
 *
 *  Date            :   November 12, 2012
 *
 *  Function        :   Main function for face detection
 *
 *  History         :
 *      12-11-12    :   Initial version.
 *
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program;  If not, see <http://www.gnu.org/licenses/>
 *
 * In other words, you are welcome to use, share and improve this program.
 * You are forbidden to forbid anyone else to use, share and improve
 * what you give them.   Happy coding!
 */

#include <stdio.h>
#include <stdlib.h>
#include "image.h"
#include "stdio-wrapper.h"
#include "haar.h"

#define INPUT_FILENAME "640.pgm"
#define OUTPUT_FILENAME "Output.pgm"

using namespace std;


int main (int argc, char *argv[]) 
{

   char *logFile, *inFile;
   int dtype_flag;

   if(argc == 4){
      inFile = argv[1];
      dtype_flag = atoi(argv[2]);
      logFile = argv[3];
   }else{
      logFile = "img.log";
      inFile = INPUT_FILENAME;
      dtype_flag = 32;                     //32 for dtype 32bit
      printf("Usage: %s <in_image> <Bitness> <logfile>\nUsing Defaults\n", argv[0]);
   }


   std::fstream olog;
   olog.open(logFile, std::fstream::in | std::fstream::out | std::fstream::app);
	
   int flag;
	
	int mode = 1;
	int i;

	/* detection parameters */
	float scaleFactor = 1.2;
	int minNeighbours = 1;


	printf("\n-- Entering main function, Params--> Bitness: %d --\r\n", dtype_flag);


	MyImage imageObj;
	MyImage *image = &imageObj;

	flag = readPgm(inFile, image);
	if (flag == -1)
	{
		printf( "Unable to open input image\n");
		return 1;
	}

	printf("-- Image Loaded- Width: %d, Height: %d\r\n", image->width, image->height);

	printf("-- Loading cascade classifier\r\n");

   MySize minSize = {20, 20};
	MySize maxSize = {0, 0};

   readTextClassifier();
	std::vector<MyRect> result;

   if(dtype_flag == 32){
	   
      myCascade cascadeObj;
	   myCascade *cascade = &cascadeObj;

	   /* classifier properties */
	   cascade->n_stages=25;
	   cascade->total_nodes=2913;
	   cascade->orig_window_size.height = 24;
	   cascade->orig_window_size.width = 24;

	   printf("\n-- Cascade Classifier Info: \n \tStages: %d\n \tTotal Filters: %d\n \tWindow Size: Width: %d, Height: %d\r\n", 
         cascade->n_stages, cascade->total_nodes, cascade->orig_window_size.width, cascade->orig_window_size.height);
      
  	   printf("\n-- Detecting faces\r\n");
	   result = detectObjects(image, minSize, maxSize, cascade, scaleFactor, minNeighbours, olog);

   }else{
	  
      myCascade16_t cascadeObj;
	   myCascade16_t *cascade = &cascadeObj;

	   /* classifier properties */
	   cascade->n_stages=25;
	   cascade->total_nodes=2913;
	   cascade->orig_window_size.height = 24;
	   cascade->orig_window_size.width = 24;

	   printf("\n-- Cascade Classifier Info: \n \tStages: %d\n \tTotal Filters: %d\n \tWindow Size: Width: %d, Height: %d\r\n", 
         cascade->n_stages, cascade->total_nodes, cascade->orig_window_size.width, cascade->orig_window_size.height);

  	   printf("\n-- Detecting faces\r\n");
	   result = detectObjects16_t(image, minSize, maxSize, cascade, scaleFactor, minNeighbours, olog);
   }

	for(i = 0; i < result.size(); i++ )
	{
		MyRect r = result[i];
		drawRectangle(image, r);
	}

	printf("\n-- Saving output --\r\n"); 
	flag = writePgm((char *)OUTPUT_FILENAME, image); 

	printf("-- Image saved --\r\n");

	/* delete image and free classifier */
	releaseTextClassifier();
	freeImage(image);
   olog.close();

	return 0;
}
