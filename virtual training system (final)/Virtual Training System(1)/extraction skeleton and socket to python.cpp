#define _CRT_SECURE_NO_WARNINGS
#define _WINSOCKAPI_


// example (2)와 동일한 코드 

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>

#include <windows.h>
#include <NuiApi.h> 
//STL
#include <iostream>

//Socket
#include<stdio.h>
#include<winsock2.h>

using namespace cv;

CvPoint points[NUI_SKELETON_POSITION_COUNT];

int createRGBImage(HANDLE h, IplImage* Color)
{
	const NUI_IMAGE_FRAME* pImageFrame = NULL;
	HRESULT hr = NuiImageStreamGetNextFrame(h, 1000, &pImageFrame);

	if (FAILED(hr))
	{
		std::cout << "Create RGB image Failed\n";
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

CvPoint SkeletonToScreen(Vector4 skeletonPoint)
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

void drawSkeleton(SOCKET sock, const NUI_SKELETON_DATA &position, IplImage* Skeleton)
{
	SOCKET clientsock;
	struct sockaddr_in clientinfo;
	int clientsize;
	clientsize = sizeof(clientinfo);

	printf("클라이언트로부터 접속을 기다리고 있습니다...\n");

	clientsock = accept(sock, (SOCKADDR*)&clientinfo, &clientsize);

	if (clientsock == INVALID_SOCKET)
		printf(" 클라이언트 소켓 연결 실패 ");
	else
		printf("연결되었습니다.");


	char skeleton_data[11 * 3 * NUI_SKELETON_POSITION_COUNT + 200] = { 0, };
	char temp_skeleton_data[15] = { 0, };

	for (int i = 0; i < NUI_SKELETON_POSITION_COUNT; ++i)
	{

		sprintf(temp_skeleton_data, "%f", position.SkeletonPositions[i].x - position.SkeletonPositions[NUI_SKELETON_POSITION_HIP_CENTER].x);
		strcat(temp_skeleton_data, ", ");
		strcat(skeleton_data, temp_skeleton_data);


		sprintf(temp_skeleton_data, "%f", -(position.SkeletonPositions[i].z - position.SkeletonPositions[NUI_SKELETON_POSITION_HIP_CENTER].z));
		strcat(temp_skeleton_data, ", ");
		strcat(skeleton_data, temp_skeleton_data);


		sprintf(temp_skeleton_data, "%f", position.SkeletonPositions[i].y - position.SkeletonPositions[NUI_SKELETON_POSITION_HIP_CENTER].y);

		if (i != NUI_SKELETON_POSITION_COUNT - 1)
			strcat(temp_skeleton_data, ", ");

		strcat(skeleton_data, temp_skeleton_data);


		points[i] = SkeletonToScreen(position.SkeletonPositions[i]);
	}
	strcat(skeleton_data, "\n");
	printf("%s", skeleton_data);

	send(clientsock, skeleton_data, sizeof(skeleton_data), 0); //소켓 보냄 (한 Frame Skeleton data)
	closesocket(clientsock);


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
void createSkeleton(SOCKET sock, HANDLE h, IplImage* Skeleton)
{
	NUI_SKELETON_FRAME skeletonFrame = { 0 };
	IplImage* Skeleton_clear = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 4);
	cvCopy(Skeleton_clear, Skeleton);
	HRESULT hr = NuiSkeletonGetNextFrame(0, &skeletonFrame);
	if (FAILED(hr))
	{
		std::cout << "골격 Frame 를 얻을수 없습니다" << std::endl;
		return;
	}
	NuiTransformSmooth(&skeletonFrame, NULL);

	for (int i = 0; i < NUI_SKELETON_COUNT; i++)
	{
		NUI_SKELETON_TRACKING_STATE state = skeletonFrame.SkeletonData[i].eTrackingState;

		if (NUI_SKELETON_TRACKED == state)
		{
			drawSkeleton(sock, skeletonFrame.SkeletonData[i], Skeleton);
		}
		cvShowImage("skeleton img", Skeleton);
	}
	cvReleaseImage(&Skeleton_clear);
}


int main(int argc, char *argv[]) {
	SOCKET sock;
	WSADATA wsa;
	struct sockaddr_in sockinfo;

	if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
		printf("초기화 실패\n");

	sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

	if (sock == INVALID_SOCKET)
		printf("소켓 생성 실패\n");

	memset(&sockinfo, 0, sizeof(sockinfo));
	sockinfo.sin_family = AF_INET;
	sockinfo.sin_port = htons(1234);
	sockinfo.sin_addr.s_addr = htonl(INADDR_ANY);
	if (bind(sock, (SOCKADDR*)&sockinfo, sizeof(sockinfo)) == SOCKET_ERROR)
		printf(" bind 실패 ");

	if (listen(sock, 5) == SOCKET_ERROR)
		printf(" 대기열 실패 ");



	HANDLE colorStreamHandle;
	HANDLE nextColorFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
	HANDLE skeletonStreamHandle = NULL;
	HANDLE nextSkeletonFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

	HRESULT hr = NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_SKELETON);


	IplImage* Color = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 4);
	IplImage* Skeleton = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 4);

	cvNamedWindow("Color img", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("skeleton img", CV_WINDOW_AUTOSIZE);


	hr = NuiImageStreamOpen(NUI_IMAGE_TYPE_COLOR, NUI_IMAGE_RESOLUTION_640x480, 0, 2, nextColorFrameEvent, &colorStreamHandle);
	if (FAILED(hr))
	{
		std::cout << "Could not open ImageStream" << std::endl;
		return hr;
	}

	hr = NuiSkeletonTrackingEnable(nextSkeletonFrameEvent, 0);
	if (FAILED(hr))
	{
		std::cout << "Could not open SkeletonStream" << std::endl;
		return hr;
	}

	while (1)
	{
		WaitForSingleObject(nextColorFrameEvent, 1000);
		createRGBImage(colorStreamHandle, Color);
		WaitForSingleObject(nextSkeletonFrameEvent, 0);
		createSkeleton(sock, skeletonStreamHandle, Skeleton);

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