/*#define _CRT_SECURE_NO_WARNINGS


#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>

#include <windows.h>
#include <NuiApi.h> 

#include <iostream>

using namespace std;
using namespace cv;

CvPoint points[NUI_SKELETON_POSITION_COUNT];

int createRGBImage(HANDLE h, IplImage* Color)
{
	const NUI_IMAGE_FRAME* pImageFrame = NULL;
	HRESULT hr = NuiImageStreamGetNextFrame(h, 1000, &pImageFrame);

	if (FAILED(hr))
	{
		cout << "Create RGB image Failed\n";
		return -1;
	}
	INuiFrameTexture* pTexture = pImageFrame->pFrameTexture;
	NUI_LOCKED_RECT LockedRect;
	pTexture->LockRect(0, &LockedRect, NULL, 0);

	if (LockedRect.Pitch != 0)
	{
		BYTE* pBuffer = (BYTE*)LockedRect.pBits;
		cvSetData(Color, pBuffer, LockedRect.Pitch);
		cvShowImage("Color Image", Color);
	}
	NuiImageStreamReleaseFrame(h, pImageFrame);

	return 0;
}

CvPoint SkeletonToScreen(FILE* file, Vector4 skeletonPoint)
{
	LONG x, y;
	USHORT depth;

	NuiTransformSkeletonToDepthImage(skeletonPoint, &x, &y, &depth, NUI_IMAGE_RESOLUTION_640x480);

	float screenPointX = static_cast<float>(x);
	float screenPointY = static_cast<float>(y);

	return cvPoint(screenPointX, screenPointY);

}

void drawBone(const NUI_SKELETON_DATA &position, NUI_SKELETON_POSITION_INDEX j1, NUI_SKELETON_POSITION_INDEX j2, IplImage *Skeleton)
{
	NUI_SKELETON_POSITION_TRACKING_STATE j1state = position.eSkeletonPositionTrackingState[j1];
	NUI_SKELETON_POSITION_TRACKING_STATE j2state = position.eSkeletonPositionTrackingState[j2];

	if (j1state == NUI_SKELETON_POSITION_TRACKED && j2state == NUI_SKELETON_POSITION_TRACKED)
	{
		cvLine(Skeleton, points[j1], points[j2], RGB(0, 255, 0), 1, 8, 0);
	}

}

// extraction skeleton 에서 변경한 부분.
void drawSkeleton(FILE* file, const NUI_SKELETON_DATA &position, IplImage* Skeleton)
{
	 // skeleton 관절부위 20개를 csv file에 저장. (이때 원점을 배꼽위치로 i=0 으로 잡아 변환 시켜준다.)
	for (int i = 0; i < NUI_SKELETON_POSITION_COUNT; ++i)
	{	
		//python의 vis 코드에 맞게 좌표는 임의적으로 변경했으며 저장된거는 xyz순으로 생각한다.
		fprintf(file, "%f,%f,%f", position.SkeletonPositions[i].x - position.SkeletonPositions[NUI_SKELETON_POSITION_HIP_CENTER].x,
			-(position.SkeletonPositions[i].z - position.SkeletonPositions[NUI_SKELETON_POSITION_HIP_CENTER].z),
			position.SkeletonPositions[i].y - position.SkeletonPositions[NUI_SKELETON_POSITION_HIP_CENTER].y); //xyz매핑 -> x, -depth, y 

		points[i] = SkeletonToScreen(file, position.SkeletonPositions[i]); 
		if (i != NUI_SKELETON_POSITION_COUNT - 1)
		{
			fprintf(file, ",");

		}
	}
	fprintf(file, "\n"); // 한 frame의 마지막을 \n으로 알린다. 

	drawBone(position, NUI_SKELETON_POSITION_HEAD, NUI_SKELETON_POSITION_SHOULDER_CENTER, Skeleton); 
	drawBone(position, NUI_SKELETON_POSITION_SHOULDER_CENTER, NUI_SKELETON_POSITION_SHOULDER_RIGHT, Skeleton);
	drawBone(position, NUI_SKELETON_POSITION_SHOULDER_RIGHT, NUI_SKELETON_POSITION_ELBOW_RIGHT, Skeleton);
	drawBone(position, NUI_SKELETON_POSITION_ELBOW_RIGHT, NUI_SKELETON_POSITION_WRIST_RIGHT, Skeleton);
	drawBone(position, NUI_SKELETON_POSITION_WRIST_RIGHT, NUI_SKELETON_POSITION_HAND_RIGHT, Skeleton);

	drawBone(position, NUI_SKELETON_POSITION_SHOULDER_CENTER, NUI_SKELETON_POSITION_SHOULDER_LEFT, Skeleton);
	drawBone(position, NUI_SKELETON_POSITION_SHOULDER_LEFT, NUI_SKELETON_POSITION_ELBOW_LEFT, Skeleton);
	drawBone(position, NUI_SKELETON_POSITION_ELBOW_LEFT, NUI_SKELETON_POSITION_WRIST_LEFT, Skeleton);
	drawBone(position, NUI_SKELETON_POSITION_WRIST_LEFT, NUI_SKELETON_POSITION_HAND_LEFT, Skeleton);

	drawBone(position, NUI_SKELETON_POSITION_SHOULDER_CENTER, NUI_SKELETON_POSITION_SPINE, Skeleton);
	drawBone(position, NUI_SKELETON_POSITION_SPINE, NUI_SKELETON_POSITION_HIP_CENTER, Skeleton);

	drawBone(position, NUI_SKELETON_POSITION_HIP_CENTER, NUI_SKELETON_POSITION_HIP_RIGHT, Skeleton);
	drawBone(position, NUI_SKELETON_POSITION_HIP_RIGHT, NUI_SKELETON_POSITION_KNEE_RIGHT, Skeleton);
	drawBone(position, NUI_SKELETON_POSITION_KNEE_RIGHT, NUI_SKELETON_POSITION_ANKLE_RIGHT, Skeleton);
	drawBone(position, NUI_SKELETON_POSITION_ANKLE_RIGHT, NUI_SKELETON_POSITION_FOOT_RIGHT, Skeleton);

	drawBone(position, NUI_SKELETON_POSITION_HIP_CENTER, NUI_SKELETON_POSITION_HIP_LEFT, Skeleton);
	drawBone(position, NUI_SKELETON_POSITION_HIP_LEFT, NUI_SKELETON_POSITION_KNEE_LEFT, Skeleton);
	drawBone(position, NUI_SKELETON_POSITION_KNEE_LEFT, NUI_SKELETON_POSITION_ANKLE_LEFT, Skeleton);
	drawBone(position, NUI_SKELETON_POSITION_ANKLE_LEFT, NUI_SKELETON_POSITION_FOOT_LEFT, Skeleton);


}
void createSkeleton(FILE* file, HANDLE h, IplImage* Skeleton)
{
	NUI_SKELETON_FRAME skeletonFrame = { 0 };
	IplImage* Skeleton_clear = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 4);
	cvCopy(Skeleton_clear, Skeleton);
	HRESULT hr = NuiSkeletonGetNextFrame(0, &skeletonFrame);
	if (FAILED(hr))
	{
		cout << "골격 Frame 를 얻을수 없습니다" << endl;
		return;
	}
	NuiTransformSmooth(&skeletonFrame, NULL);

	for (int i = 0; i < NUI_SKELETON_COUNT; i++)

	{
		NUI_SKELETON_TRACKING_STATE state = skeletonFrame.SkeletonData[i].eTrackingState;

		if (NUI_SKELETON_TRACKED == state)
		{
			drawSkeleton(file, skeletonFrame.SkeletonData[i], Skeleton);
		}
		cvShowImage("skeleton img", Skeleton);
	}
	cvReleaseImage(&Skeleton_clear);
}


int main() {


	FILE* file;
	file = fopen("test.csv", "w");

	HANDLE colorStreamHandle;
	HANDLE nextColorFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
	HANDLE skeletonStreamHandle = NULL; //초기화 안됬다고 계속 그래서;;
	HANDLE nextSkeletonFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

	HRESULT hr = NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_SKELETON); //color와 skeleton을 사용하겠다. 플래그를 주는것


	IplImage* Color = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 4);  //IplImage 형식의 이미지 파일을 선언
	IplImage* Skeleton = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 4);

	cvNamedWindow("Color img", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("skeleton img", CV_WINDOW_AUTOSIZE);


	//Tracking 기능 활성화
	hr = NuiImageStreamOpen(NUI_IMAGE_TYPE_COLOR, NUI_IMAGE_RESOLUTION_640x480, 0, 2, nextColorFrameEvent, &colorStreamHandle); // rgb 스트림오픈
	if (FAILED(hr))
	{
		cout << "Could not open ImageStream" << endl;
		return hr;
	}

	hr = NuiSkeletonTrackingEnable(nextSkeletonFrameEvent, 0); // 마찬가지로 오픈
	if (FAILED(hr))
	{
		cout << "Could not open SkeletonStream" << endl;
		return hr;
	}

	while (1)
	{
		WaitForSingleObject(nextColorFrameEvent, 1000); 
		createRGBImage(colorStreamHandle, Color); 
		WaitForSingleObject(nextSkeletonFrameEvent, 0);
		createSkeleton(file, skeletonStreamHandle, Skeleton);

		if (cvWaitKey(10) == 0x001b)
		{
			break;
		}
	}
	NuiShutdown();

	cvReleaseImageHeader(&Color);

	cvDestroyAllWindows();
	fclose(file);

	return 0;
}*/