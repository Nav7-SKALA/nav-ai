from datetime import datetime
import time
import functools
import asyncio

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        self.logs = []
        self.start_time = None
        
    def start_monitoring(self):
        """전체 모니터링 시작"""
        self.start_time = time.time()
        self.logs = []
        print(f"🚀 성능 모니터링 시작 - {datetime.now().strftime('%H:%M:%S')}")
        
    def log_step(self, step_name, duration, status="✅"):
        """단계별 로그 기록"""
        total_elapsed = time.time() - self.start_time if self.start_time else 0
        log_entry = {
            'step': step_name,
            'duration': duration,
            'total_elapsed': total_elapsed,
            'status': status,
            'timestamp': datetime.now().strftime('%H:%M:%S.%f')[:-3]
        }
        self.logs.append(log_entry)
        print(f"{status} {step_name}: {duration:.2f}초 (총 {total_elapsed:.2f}초)")
        
    def print_summary(self):
        """성능 요약 출력"""
        print("\n" + "="*60)
        print("📊 성능 분석 요약")
        print("="*60)
        
        total_time = self.logs[-1]['total_elapsed'] if self.logs else 0
        
        for log in self.logs:
            percentage = (log['duration'] / total_time * 100) if total_time > 0 else 0
            print(f"{log['status']} {log['step']:<25} | {log['duration']:>6.2f}초 ({percentage:>5.1f}%)")
            
        print(f"\n🏁 총 실행 시간: {total_time:.2f}초")
        
        # 가장 느린 단계 찾기
        if self.logs:
            slowest = max(self.logs, key=lambda x: x['duration'])
            print(f"🐌 가장 느린 단계: {slowest['step']} ({slowest['duration']:.2f}초)")

# 전역 모니터 인스턴스
monitor = PerformanceMonitor()

def performance_tracker(step_name=None):
    """성능 추적 데코레이터"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = step_name or func.__name__
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                monitor.log_step(name, duration, "✅")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                monitor.log_step(name, duration, "❌")
                print(f"❌ {name} 실패: {e}")
                raise
                
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            name = step_name or func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                monitor.log_step(name, duration, "✅")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                monitor.log_step(name, duration, "❌")
                print(f"❌ {name} 실패: {e}")
                raise
        
        # 함수가 코루틴인지 확인
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator