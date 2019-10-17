//기본 골격좌표를 얻어 화면에 vis하는 코드

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>

#include <windows.h>
#include <NuiApi.h> // Microsoft Kinect SDK


//STL
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

CvPoint points[NUI_SKELETON_POSITION_COUNT];

//RGB img 계속받아오는
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

// 좌표변환 : kinect의 기본 상대좌표에서 절대좌표로 변경해 화면에 골격그림을 보여주기위해.
CvPoint SkeletonToScreen(Vector4 skeletonPoint)
{
	LONG x, y;
	USHORT depth;

	NuiTransformSkeletonToDepthImage(skeletonPoint, &x, &y, &depth, NUI_IMAGE_RESOLUTION_640x480);

	float screenPointX = static_cast<float>(x);
	float screenPointY = static_cast<float>(y);

	return cvPoint(screenPointX, screenPointY);

}

// 골격좌표를 이용해서 그림을 그리기위한.
void drawBone(const NUI_SKELETON_DATA &position, NUI_SKELETON_POSITION_INDEX j1, NUI_SKELETON_POSITION_INDEX j2, IplImage *Skeleton)
{
	NUI_SKELETON_POSITION_TRACKING_STATE j1state = position.eSkeletonPositionTrackingState[j1];
	NUI_SKELETON_POSITION_TRACKING_STATE j2state = position.eSkeletonPositionTrackingState[j2];

	if (j1state == NUI_SKELETON_POSITION_TRACKED && j2state == NUI_SKELETON_POSITION_TRACKED)
	{
		cvLine(Skeleton, points[j1], points[j2], RGB(0, 255, 0), 1, 8, 0);
	}

}


void drawSkeleton(const NUI_SKELETON_DATA &position, IplImage* Skeleton)
{
	for (int i = 0; i < NUI_SKELETON_POSITION_COUNT; ++i)
	{
		points[i] = SkeletonToScreen(position.SkeletonPositions[i]);
	}

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

// 골격좌표 계속 받아오는
void createSkeleton(HANDLE h, IplImage* Skeleton)
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
			drawSkeleton(skeletonFrame.SkeletonData[i], Skeleton);
		}
		cvShowImage("skeleton img", Skeleton);
	}
	cvReleaseImage(&Skeleton_clear);
}





int main() {

	HANDLE colorStreamHandle;
	HANDLE nextColorFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
	HANDLE skeletonStreamHandle = NULL; //초기화 안됬다고 계속 그래서;;
	HANDLE nextSkeletonFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

	HRESULT hr = NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_SKELETON); //color와 skeleton을 사용하겠다. 플래그


	IplImage* Color = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 4);  //IplImage 형식의 이미지 파일을 선언
	IplImage* Skeleton = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 4);

	cvNamedWindow("Color img", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("skeleton img", CV_WINDOW_AUTOSIZE);


	//RGB 스트림 개방
	hr = NuiImageStreamOpen(NUI_IMAGE_TYPE_COLOR, NUI_IMAGE_RESOLUTION_640x480, 0, 2, nextColorFrameEvent, &colorStreamHandle);
	if (FAILED(hr))
	{
		cout << "Could not open ImageStream" << endl;
		return hr;
	}

	// 골격 트랙킹 
	hr = NuiSkeletonTrackingEnable(nextSkeletonFrameEvent, 0);
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
		createSkeleton(skeletonStreamHandle, Skeleton);

		if (cvWaitKey(10) == 0x001b)
		{
			break;
		}
	}
	NuiShutdown();

	cvReleaseImageHeader(&Color);

	cvDestroyAllWindows();
	return 0;
}